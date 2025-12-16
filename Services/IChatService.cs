using LocalLLMChat.Models;

namespace LocalLLMChat.Services;

public interface IChatService
{
    Task<ChatResponse> ProcessMessageAsync(ChatRequest request, CancellationToken cancellationToken = default);
    IAsyncEnumerable<string> ProcessMessageStreamingAsync(ChatRequest request, CancellationToken cancellationToken = default);
    ChatSession? GetSession(string sessionId);
    ChatSession CreateSession();
    void ClearSession(string sessionId);
}
