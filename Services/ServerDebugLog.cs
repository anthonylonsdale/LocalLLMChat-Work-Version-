using System.Collections.Concurrent;

namespace LocalLLMChat.Services;

public interface IServerDebugLog
{
    void Add(string source, string message);
    IReadOnlyList<DebugLogEntry> GetEntries(int max = 200);
}

public class DebugLogEntry
{
    public long Id { get; init; }
    public DateTimeOffset Timestamp { get; init; }
    public string Source { get; init; } = string.Empty;
    public string Message { get; init; } = string.Empty;
}

public class ServerDebugLog : IServerDebugLog
{
    private readonly ConcurrentQueue<DebugLogEntry> _entries = new();
    private long _nextId;
    private const int MaxEntries = 500;

    public void Add(string source, string message)
    {
        var entry = new DebugLogEntry
        {
            Id = Interlocked.Increment(ref _nextId),
            Timestamp = DateTimeOffset.UtcNow,
            Source = source,
            Message = message
        };

        _entries.Enqueue(entry);

        while (_entries.Count > MaxEntries && _entries.TryDequeue(out _))
        {
        }
    }

    public IReadOnlyList<DebugLogEntry> GetEntries(int max = 200)
    {
        var window = Math.Max(1, max);
        return _entries
            .Skip(Math.Max(0, _entries.Count - window))
            .ToList();
    }
}
