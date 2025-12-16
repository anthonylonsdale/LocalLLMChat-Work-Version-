using LocalLLMChat.Models;
using System.Text;

namespace LocalLLMChat.Services.Plugins;

/// <summary>
/// Plugin that manages conversation history for context continuity
/// </summary>
public class ConversationHistoryPlugin : IChatPlugin
{
    public string Name => "ConversationHistory";
    public int Priority => 20;
    public bool IsEnabled { get; set; } = true;

    /// <summary>
    /// Maximum number of previous messages to include in context
    /// </summary>
    public int MaxHistoryMessages { get; set; } = 10;

    public Task<PluginContext> ProcessBeforeInferenceAsync(PluginContext context, CancellationToken cancellationToken = default)
    {
        if (!IsEnabled) return Task.FromResult(context);

        var sb = new StringBuilder();

        // Get recent conversation history
        var recentMessages = context.Session.Messages
            .TakeLast(MaxHistoryMessages)
            .ToList();

        // Format conversation history
        foreach (var message in recentMessages)
        {
            var role = message.Role.Equals("user", StringComparison.OrdinalIgnoreCase) ? "User" : "Assistant";
            sb.AppendLine($"{role}: {message.Content}");
        }

        // Add current user message
        sb.AppendLine($"User: {context.UserMessage}");
        sb.AppendLine("Assistant:");

        // Append to processed prompt (after system prompt if present)
        if (context.ProcessedPrompt.Contains("System:"))
        {
            context.ProcessedPrompt = context.ProcessedPrompt + "\n" + sb.ToString();
        }
        else
        {
            context.ProcessedPrompt = sb.ToString();
        }

        return Task.FromResult(context);
    }

    public Task<PluginContext> ProcessAfterInferenceAsync(PluginContext context, CancellationToken cancellationToken = default)
    {
        if (!IsEnabled) return Task.FromResult(context);

        // Add user message to history
        context.Session.Messages.Add(new ChatMessage
        {
            Role = "user",
            Content = context.UserMessage,
            Timestamp = DateTime.UtcNow
        });

        // Add assistant response to history
        context.Session.Messages.Add(new ChatMessage
        {
            Role = "assistant",
            Content = context.Response,
            Timestamp = DateTime.UtcNow
        });

        context.Session.LastActivity = DateTime.UtcNow;

        return Task.FromResult(context);
    }
}
