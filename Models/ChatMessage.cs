namespace LocalLLMChat.Models;

public class ChatMessage
{
    public string Role { get; set; } = "user";
    public string Content { get; set; } = string.Empty;
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
}

public class ChatRequest
{
    public string Message { get; set; } = string.Empty;
    public string? SessionId { get; set; }
    public bool UseRag { get; set; }
}

public class ChatResponse
{
    public string Response { get; set; } = string.Empty;
    public string SessionId { get; set; } = string.Empty;
    public bool Success { get; set; }
    public string? Error { get; set; }
}

public class ChatSession
{
    public string SessionId { get; set; } = Guid.NewGuid().ToString();
    public List<ChatMessage> Messages { get; set; } = new();
    public DateTime Created { get; set; } = DateTime.UtcNow;
    public DateTime LastActivity { get; set; } = DateTime.UtcNow;
}
