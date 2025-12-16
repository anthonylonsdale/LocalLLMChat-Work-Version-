namespace LocalLLMChat.Rag;

public class RagResult
{
    public string Answer { get; set; } = string.Empty;
    public List<RagSource> Sources { get; set; } = new();
    public string? DebugContextPreview { get; set; }
}

public class RagSource
{
    public string SourceId { get; set; } = string.Empty;
    public int ChunkIndex { get; set; }
    public string SourcePath { get; set; } = string.Empty;
    public float Score { get; set; }
}
