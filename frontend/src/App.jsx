import { useCallback, useState } from 'react';
import { LiveKitRoom, RoomAudioRenderer } from '@livekit/components-react';
import StartScreen from './components/StartScreen.jsx';
import Stage from './components/Stage.jsx';
import ControlBar from './components/ControlBar.jsx';
import Transcript from './components/Transcript.jsx';

export default function App() {
  const [session, setSession] = useState(null);
  const [error, setError] = useState(null);
  const [joining, setJoining] = useState(false);

  const startCall = useCallback(async () => {
    setJoining(true);
    setError(null);
    try {
      const res = await fetch('/token', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      });
      const body = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(body.detail || `token request failed (${res.status})`);
      }
      setSession(body);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setJoining(false);
    }
  }, []);

  const endCall = useCallback(() => setSession(null), []);

  if (!session) {
    return <StartScreen onStart={startCall} joining={joining} error={error} />;
  }

  return (
    <LiveKitRoom
      serverUrl={session.url}
      token={session.token}
      audio={{ echoCancellation: true, noiseSuppression: true }}
      video={false}
      connect
      onDisconnected={endCall}
      onError={(err) => setError(err.message)}
    >
      <div className="app-shell">
        <header className="app-header">
          <span className="wordmark">Kilos</span>
          <span className="room-pill">{session.room}</span>
        </header>

        {error && (
          <div className="banner error" role="alert">
            {error}
          </div>
        )}

        <main className="app-main">
          <Stage avatarExpected={session.avatar} />
          <Transcript />
        </main>

        <ControlBar />
        <RoomAudioRenderer />
      </div>
    </LiveKitRoom>
  );
}
