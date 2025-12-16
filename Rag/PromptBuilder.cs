using System.Text;

namespace LocalLLMChat.Rag;

public class PromptBuilder
{
    private readonly RagConfig _config;

    public PromptBuilder(RagConfig config)
    {
        _config = config;
    }

    public string Build(string question, IReadOnlyList<(DocumentChunk chunk, float score)> contexts)
    {
        var sb = new StringBuilder();
        sb.AppendLine("SYSTEM:");
        sb.AppendLine("You are a helpful assistant. Use ONLY the provided context when answering. If the context is insufficient, say you do not know.");
        sb.AppendLine();
        sb.AppendLine("CONTEXT:");

        var total = 0;
        foreach (var (chunk, score) in contexts)
        {
            var next = chunk.Text ?? string.Empty;
            var candidateLength = total + next.Length;
            if (candidateLength > _config.MaxContextChars) break;
            total = candidateLength;

            sb.AppendLine($"[Source: {chunk.SourceId} #{chunk.ChunkIndex}]");
            sb.AppendLine(next);
            sb.AppendLine();
        }

        sb.AppendLine("USER:");
        sb.AppendLine(question);
        sb.AppendLine();
        sb.AppendLine("REQUIRE:");
        sb.AppendLine("- Be concise and factual.");
        sb.AppendLine("- If you used any context, list Sources with SourceId and chunk index.");

        return sb.ToString().Trim();
    }
}
