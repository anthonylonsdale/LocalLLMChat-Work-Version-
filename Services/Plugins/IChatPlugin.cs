using LocalLLMChat.Models;

namespace LocalLLMChat.Services.Plugins;

/// <summary>
/// Interface for chatbot plugins that can process and modify chat interactions
/// </summary>
public interface IChatPlugin
{
    /// <summary>
    /// Plugin name for identification
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Plugin priority (lower = earlier execution)
    /// </summary>
    int Priority { get; }

    /// <summary>
    /// Whether the plugin is enabled
    /// </summary>
    bool IsEnabled { get; set; }

    /// <summary>
    /// Called before the prompt is sent to the LLM
    /// </summary>
    Task<PluginContext> ProcessBeforeInferenceAsync(PluginContext context, CancellationToken cancellationToken = default);

    /// <summary>
    /// Called after the LLM generates a response
    /// </summary>
    Task<PluginContext> ProcessAfterInferenceAsync(PluginContext context, CancellationToken cancellationToken = default);
}

/// <summary>
/// Context passed through the plugin pipeline
/// </summary>
public class PluginContext
{
    public string UserMessage { get; set; } = string.Empty;
    public string ProcessedPrompt { get; set; } = string.Empty;
    public string Response { get; set; } = string.Empty;
    public ChatSession Session { get; set; } = new();
    public Dictionary<string, object> Metadata { get; set; } = new();

    /// <summary>
    /// Flag to stop further plugin processing
    /// </summary>
    public bool StopProcessing { get; set; }
}
