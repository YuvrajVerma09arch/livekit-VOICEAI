export default function StartScreen({ onStart, joining, error }) {
  return (
    <div className="start-screen">
      <div className="start-card">
        <div className="start-orb" />
        <h1>Kilos</h1>
        <p>A LiveKit voice agent with a LiveAvatar embodiment.</p>
        <button className="primary" onClick={onStart} disabled={joining}>
          {joining ? 'Connecting…' : 'Start call'}
        </button>
        {error && <p className="error">{error}</p>}
        <p className="hint">
          The token server must be running on port 7860 (<code>uvicorn token_server:app</code>).
        </p>
      </div>
    </div>
  );
}
