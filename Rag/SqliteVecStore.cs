using Microsoft.Data.Sqlite;
using System.Text.Json;

namespace LocalLLMChat.Rag;

public class SqliteVecStore
{
    private readonly string _dbPath;
    private const string TableName = "rag_chunks";

    public SqliteVecStore(RagConfig config)
    {
        _dbPath = ResolvePath(config.SqliteDbPath);
        var dir = Path.GetDirectoryName(_dbPath);
        if (!string.IsNullOrEmpty(dir))
        {
            Directory.CreateDirectory(dir);
        }
    }

    public async Task EnsureCreatedAsync(CancellationToken cancellationToken = default)
    {
        await using var conn = CreateConnection();
        await conn.OpenAsync(cancellationToken);

        var cmd = conn.CreateCommand();
        cmd.CommandText = $@"
CREATE TABLE IF NOT EXISTS {TableName} (
    id TEXT PRIMARY KEY,
    sourceId TEXT,
    sourcePath TEXT,
    chunkIndex INTEGER,
    text TEXT,
    ingestedAt TEXT,
    embedding TEXT
);
";
        await cmd.ExecuteNonQueryAsync(cancellationToken);
    }

    public async Task UpsertAsync(IEnumerable<DocumentChunk> chunks, IEnumerable<IReadOnlyList<float>> vectors, CancellationToken cancellationToken = default)
    {
        var chunkList = chunks.ToList();
        var vectorList = vectors.ToList();
        if (chunkList.Count != vectorList.Count) throw new InvalidOperationException("Chunks and vectors length mismatch.");

        await using var conn = CreateConnection();
        await conn.OpenAsync(cancellationToken);
        await using var tx = await conn.BeginTransactionAsync(cancellationToken);

        for (int i = 0; i < chunkList.Count; i++)
        {
            var chunk = chunkList[i];
            var vec = vectorList[i];
            var cmd = conn.CreateCommand();
            cmd.CommandText = $@"
INSERT INTO {TableName} (id, sourceId, sourcePath, chunkIndex, text, ingestedAt, embedding)
VALUES ($id,$sid,$spath,$idx,$text,$ing,$emb)
ON CONFLICT(id) DO UPDATE SET
    sourceId=excluded.sourceId,
    sourcePath=excluded.sourcePath,
    chunkIndex=excluded.chunkIndex,
    text=excluded.text,
    ingestedAt=excluded.ingestedAt,
    embedding=excluded.embedding;
";
            cmd.Parameters.AddWithValue("$id", chunk.ChunkId);
            cmd.Parameters.AddWithValue("$sid", chunk.SourceId);
            cmd.Parameters.AddWithValue("$spath", chunk.SourcePath);
            cmd.Parameters.AddWithValue("$idx", chunk.ChunkIndex);
            cmd.Parameters.AddWithValue("$text", chunk.Text);
            cmd.Parameters.AddWithValue("$ing", chunk.IngestedAt.UtcDateTime.ToString("o"));
            cmd.Parameters.AddWithValue("$emb", JsonSerializer.Serialize(vec));
            await cmd.ExecuteNonQueryAsync(cancellationToken);
        }

        await tx.CommitAsync(cancellationToken);
    }

    public async Task<IReadOnlyList<(DocumentChunk chunk, float score)>> SearchAsync(IReadOnlyList<float> queryVector, int topK, CancellationToken cancellationToken = default)
    {
        await using var conn = CreateConnection();
        await conn.OpenAsync(cancellationToken);

        var cmd = conn.CreateCommand();
        cmd.CommandText = $@"SELECT id, sourceId, sourcePath, chunkIndex, text, ingestedAt, embedding FROM {TableName};";

        var results = new List<(DocumentChunk chunk, float score)>();
        await using var reader = await cmd.ExecuteReaderAsync(cancellationToken);
        while (await reader.ReadAsync(cancellationToken))
        {
            var embJson = reader.GetString(6);
            var emb = JsonSerializer.Deserialize<List<float>>(embJson) ?? new List<float>();
            var score = CosineSimilarity(queryVector, emb);

            var chunk = new DocumentChunk
            {
                ChunkId = reader.GetString(0),
                SourceId = reader.GetString(1),
                SourcePath = reader.GetString(2),
                ChunkIndex = reader.GetInt32(3),
                Text = reader.GetString(4),
                IngestedAt = DateTimeOffset.Parse(reader.GetString(5))
            };
            results.Add((chunk, score));
        }

        return results
            .OrderByDescending(r => r.score)
            .Take(topK)
            .ToList();
    }

    private static float CosineSimilarity(IReadOnlyList<float> a, IReadOnlyList<float> b)
    {
        if (a.Count != b.Count || a.Count == 0) return 0f;
        double dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < a.Count; i++)
        {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        if (normA == 0 || normB == 0) return 0f;
        return (float)(dot / (Math.Sqrt(normA) * Math.Sqrt(normB)));
    }

    private SqliteConnection CreateConnection()
    {
        return new SqliteConnection($"Data Source={_dbPath};");
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
