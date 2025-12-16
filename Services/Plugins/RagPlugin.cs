using System.Text;

namespace LocalLLMChat.Services.Plugins;

/// <summary>
/// Plugin that integrates RAG (Retrieval-Augmented Generation) for context enhancement
/// </summary>
public class RagPlugin : IChatPlugin
{
    private readonly IRagService _ragService;
    private readonly ILogger<RagPlugin> _logger;

    public string Name => "RAG";
    public int Priority => 15; // After system prompt, before conversation history
    public bool IsEnabled { get; set; } = true;

    /// <summary>
    /// Number of documents to retrieve
    /// </summary>
    public int TopK { get; set; } = 3;

    /// <summary>
    /// Minimum relevance score for including a document
    /// </summary>
    public float MinRelevanceScore { get; set; } = 0.1f;

    public RagPlugin(IRagService ragService, ILogger<RagPlugin> logger)
    {
        _ragService = ragService;
        _logger = logger;
    }

    public async Task<PluginContext> ProcessBeforeInferenceAsync(PluginContext context, CancellationToken cancellationToken = default)
    {
        if (!IsEnabled) return context;

        try
        {
            // Search for relevant documents
            var documents = await _ragService.SearchAsync(context.UserMessage, TopK, cancellationToken);
            var relevantDocs = documents.Where(d => d.RelevanceScore >= MinRelevanceScore).ToList();

            if (relevantDocs.Any())
            {
                _logger.LogInformation("Found {Count} relevant documents for query", relevantDocs.Count);

                var contextBuilder = new StringBuilder();
                contextBuilder.AppendLine("Relevant Context:");
                contextBuilder.AppendLine("---");

                foreach (var doc in relevantDocs)
                {
                    contextBuilder.AppendLine(doc.Content);
                    contextBuilder.AppendLine();
                }

                contextBuilder.AppendLine("---");
                contextBuilder.AppendLine("Use the above context to help answer the user's question if relevant.");
                contextBuilder.AppendLine();

                // Insert RAG context after system prompt but before conversation
                if (context.ProcessedPrompt.Contains("System:"))
                {
                    var systemEnd = context.ProcessedPrompt.IndexOf("\n\n", context.ProcessedPrompt.IndexOf("System:"));
                    if (systemEnd > 0)
                    {
                        context.ProcessedPrompt = context.ProcessedPrompt.Insert(systemEnd + 2, contextBuilder.ToString());
                    }
                    else
                    {
                        context.ProcessedPrompt += "\n\n" + contextBuilder.ToString();
                    }
                }
                else
                {
                    context.ProcessedPrompt = contextBuilder.ToString() + context.ProcessedPrompt;
                }

                // Store retrieved documents in metadata for later use
                context.Metadata["rag_documents"] = relevantDocs;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error during RAG processing");
            // Continue without RAG context on error
        }

        return context;
    }

    public Task<PluginContext> ProcessAfterInferenceAsync(PluginContext context, CancellationToken cancellationToken = default)
    {
        // No post-processing needed for RAG
        return Task.FromResult(context);
    }
}
