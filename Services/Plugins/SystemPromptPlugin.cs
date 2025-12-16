namespace LocalLLMChat.Services.Plugins;

/// <summary>
/// Plugin that adds system prompt/instructions to the conversation
/// </summary>
public class SystemPromptPlugin : IChatPlugin
{
    private readonly ILlmService _llmService;

    public string Name => "SystemPrompt";
    public int Priority => 10;
    public bool IsEnabled { get; set; } = true;

    public SystemPromptPlugin(ILlmService llmService)
    {
        _llmService = llmService;
    }

    public Task<PluginContext> ProcessBeforeInferenceAsync(PluginContext context, CancellationToken cancellationToken = default)
    {
        if (!IsEnabled) return Task.FromResult(context);

        var systemPrompt = _llmService.RuntimeSettings.SystemPrompt;

        // Prepend system prompt if not already in the processed prompt
        if (!context.ProcessedPrompt.Contains("System:") && !string.IsNullOrEmpty(systemPrompt))
        {
            context.ProcessedPrompt = $"System: {systemPrompt}\n\n{context.ProcessedPrompt}";
        }

        return Task.FromResult(context);
    }

    public Task<PluginContext> ProcessAfterInferenceAsync(PluginContext context, CancellationToken cancellationToken = default)
    {
        // No post-processing needed for system prompt
        return Task.FromResult(context);
    }
}
