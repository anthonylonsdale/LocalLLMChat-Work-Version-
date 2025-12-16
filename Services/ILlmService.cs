using LocalLLMChat.Models;

namespace LocalLLMChat.Services;

public interface ILlmService
{
    Task<string> GenerateResponseAsync(string prompt, CancellationToken cancellationToken = default);
    IAsyncEnumerable<string> GenerateStreamingResponseAsync(string prompt, CancellationToken cancellationToken = default);
    bool IsModelLoaded { get; }
    Task LoadModelAsync();
    LlmSettings Settings { get; }
    RuntimeLlmSettings RuntimeSettings { get; }
    void UpdateRuntimeSettings(RuntimeLlmSettings newSettings);
    Task<SettingsUpdateResult> ApplySettingsAsync(LlmSettings settings, CancellationToken cancellationToken = default);
}

public class SettingsUpdateResult
{
    public bool Reloaded { get; set; }
    public LlmSettings Settings { get; set; } = new();
}
