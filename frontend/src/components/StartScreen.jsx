import { useState } from 'react';

export default function StartScreen({ onStart, joining, error }) {
  const [password, setPassword] = useState('');

  const submit = (e) => {
    e.preventDefault();
    onStart(password);
  };

  return (
    <div className="start-screen">
      <div className="start-card">
        <div className="start-orb" />
        <h1>Kilos</h1>
        <p>A LiveKit voice agent with a LiveAvatar embodiment.</p>
        <form onSubmit={submit}>
          <input
            type="password"
            placeholder="Access password"
            autoComplete="current-password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
          <button className="primary" type="submit" disabled={joining}>
            {joining ? 'Connecting…' : 'Start call'}
          </button>
        </form>
        {error && <p className="error">{error}</p>}
        <p className="hint">
          The token server must be running on port 7860 (<code>uvicorn backend.token.app:app</code>).
        </p>
      </div>
    </div>
  );
}
