import {
  BarVisualizer,
  VideoTrack,
  useStartAudio,
  useVoiceAssistant,
} from '@livekit/components-react';

const STATE_LABEL = {
  disconnected: 'offline',
  connecting: 'connecting',
  'pre-connect-buffering': 'buffering',
  initializing: 'starting',
  idle: 'ready',
  listening: 'listening',
  thinking: 'thinking',
  speaking: 'speaking',
  failed: 'error',
};

export default function Stage({ avatarExpected }) {
  const { state, videoTrack, audioTrack } = useVoiceAssistant();
  const { mergedProps, canPlayAudio } = useStartAudio({
    props: { className: 'start-audio' },
  });

  const label = STATE_LABEL[state] ?? state ?? 'connecting';
  const waiting = Boolean(avatarExpected && !videoTrack);

  return (
    <section className="stage" data-state={state} data-waiting={waiting || undefined}>
      <div className="stage-frame">
        {videoTrack ? (
          <VideoTrack trackRef={videoTrack} className="avatar-video" />
        ) : (
          <div className="visualizer-wrap">
            <div className={`avatar-silhouette${waiting ? ' is-waiting' : ''}`}>
              <span className="silhouette-ring" />
              <span className="silhouette-head" />
              <span className="silhouette-body" />
            </div>

            <BarVisualizer
              state={state}
              track={audioTrack}
              barCount={7}
              options={{ minHeight: 20, maxHeight: 100 }}
              className="visualizer"
            />

            <div className={`avatar-waiting${waiting ? ' is-visible' : ''}`}>
              <span className="spinner" />
              <span>Kilos is waking up…</span>
            </div>
          </div>
        )}

        <div className="state-badge" data-state={state}>
          <span className="dot" />
          {label}
        </div>

        {!canPlayAudio && (
          <button {...mergedProps} className="start-audio" style={{ display: 'block' }}>
            Enable sound
          </button>
        )}
      </div>
    </section>
  );
}
