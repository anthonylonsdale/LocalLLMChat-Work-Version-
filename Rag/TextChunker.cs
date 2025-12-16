using System.Text;

namespace LocalLLMChat.Rag;

public class TextChunker
{
    private readonly RagConfig _config;

    public TextChunker(RagConfig config)
    {
        _config = config;
    }

    public IReadOnlyList<DocumentChunk> Chunk(string sourceId, string sourcePath, string text)
    {
        var normalized = NormalizeText(text);
        var chunks = new List<DocumentChunk>();
        var max = _config.ChunkMaxChars;
        var overlap = _config.ChunkOverlapChars;

        var paragraphs = normalized.Split(new[] { "\n\n", "\r\n\r\n" }, StringSplitOptions.RemoveEmptyEntries);
        var buffer = new StringBuilder();
        var index = 0;

        foreach (var para in paragraphs)
        {
            var trimmed = para.Trim();
            if (trimmed.Length == 0) continue;

            if (buffer.Length + trimmed.Length + 2 > max)
            {
                if (buffer.Length > 0)
                {
                    chunks.Add(CreateChunk(sourceId, sourcePath, index++, buffer.ToString()));
                    var bufferText = buffer.ToString();
                    buffer.Clear();
                    if (overlap > 0 && bufferText.Length > overlap)
                    {
                        buffer.Append(bufferText[^overlap..]);
                        buffer.AppendLine();
                    }
                }
            }

            buffer.AppendLine(trimmed);
        }

        if (buffer.Length > 0)
        {
            chunks.Add(CreateChunk(sourceId, sourcePath, index++, buffer.ToString()));
        }

        return chunks;
    }

    private static string NormalizeText(string text)
    {
        var cleaned = text.Replace("\r\n", "\n");
        while (cleaned.Contains("  "))
        {
            cleaned = cleaned.Replace("  ", " ");
        }
        return cleaned.Trim();
    }

    private static DocumentChunk CreateChunk(string sourceId, string sourcePath, int index, string text)
    {
        var idSource = $"{sourcePath}|{index}|{text.Length}";
        var chunkId = ComputeHash(idSource);
        return new DocumentChunk
        {
            ChunkId = chunkId,
            SourceId = sourceId,
            SourcePath = sourcePath,
            ChunkIndex = index,
            Text = text.Trim(),
            IngestedAt = DateTimeOffset.UtcNow
        };
    }

    private static string ComputeHash(string input)
    {
        using var sha = System.Security.Cryptography.SHA256.Create();
        var bytes = sha.ComputeHash(Encoding.UTF8.GetBytes(input));
        return Convert.ToHexString(bytes);
    }
}
