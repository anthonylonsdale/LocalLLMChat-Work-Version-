using LocalLLMChat.Services;
using LocalLLMChat.Rag.DocumentLoaders;
using System.Text;

namespace LocalLLMChat.Rag;

public class RagPipeline
{
    private readonly RagConfig _config;
    private readonly TextChunker _chunker;
    private readonly GgufEmbeddingGenerator _embedder;
    private readonly SqliteVecStore _store;
    private readonly ILlmService _llmService;
    private readonly IEnumerable<ITextDocumentLoader> _loaders;
    private readonly PromptBuilder _promptBuilder;
    private readonly ILogger<RagPipeline> _logger;

    public RagPipeline(
        RagConfig config,
        TextChunker chunker,
        GgufEmbeddingGenerator embedder,
        SqliteVecStore store,
        ILlmService llmService,
        IEnumerable<ITextDocumentLoader> loaders,
        PromptBuilder promptBuilder,
        ILogger<RagPipeline> logger)
    {
        _config = config;
        _chunker = chunker;
        _embedder = embedder;
        _store = store;
        _llmService = llmService;
        _loaders = loaders;
        _promptBuilder = promptBuilder;
        _logger = logger;
    }

    public async Task IngestAsync(CancellationToken cancellationToken = default)
    {
        var root = ResolvePath(_config.DocumentsRootPath);
        Directory.CreateDirectory(root);

        await _store.EnsureCreatedAsync(cancellationToken);

        var allChunks = new List<DocumentChunk>();
        foreach (var loader in _loaders)
        {
            foreach (var (sourceId, sourcePath, text) in loader.LoadAll(root))
            {
                var chunks = _chunker.Chunk(sourceId, sourcePath, text);
                allChunks.AddRange(chunks);
            }
        }

        if (allChunks.Count == 0)
        {
            _logger.LogWarning("No documents found under {Root}", root);
            return;
        }

        var texts = allChunks.Select(c => c.Text).ToList();
        var vectors = await _embedder.EmbedBatchAsync(texts, cancellationToken);
        await _store.UpsertAsync(allChunks, vectors, cancellationToken);
        _logger.LogInformation("Ingested {Count} chunks into vector store.", allChunks.Count);
    }

    public async Task<RagResult> AskAsync(string question, CancellationToken cancellationToken = default)
    {
        await _store.EnsureCreatedAsync(cancellationToken);

        var queryVec = await _embedder.EmbedAsync(question, cancellationToken);
        var results = await _store.SearchAsync(queryVec, _config.TopK, cancellationToken);

        var prompt = _promptBuilder.Build(question, results);
        var answer = await _llmService.GenerateResponseAsync(prompt, cancellationToken);

        var preview = BuildPreview(results);

        var sources = results.Select(r => new RagSource
        {
            SourceId = r.chunk.SourceId,
            ChunkIndex = r.chunk.ChunkIndex,
            SourcePath = r.chunk.SourcePath,
            Score = r.score
        }).ToList();

        return new RagResult
        {
            Answer = answer,
            Sources = sources,
            DebugContextPreview = preview
        };
    }

    private static string BuildPreview(IReadOnlyList<(DocumentChunk chunk, float score)> results)
    {
        var sb = new StringBuilder();
        foreach (var (chunk, score) in results)
        {
            sb.AppendLine($"[{chunk.SourceId} #{chunk.ChunkIndex}] (score {score:F4})");
            sb.AppendLine(chunk.Text);
            sb.AppendLine();
        }
        return sb.ToString();
    }

    private static string ResolvePath(string path)
    {
        if (path.StartsWith("~"))
        {
            var root = AppDomain.CurrentDomain.BaseDirectory.TrimEnd(Path.DirectorySeparatorChar);
            return Path.Combine(root, path.TrimStart('~', '/', '\\'));
        }
        return path;
    }
}
