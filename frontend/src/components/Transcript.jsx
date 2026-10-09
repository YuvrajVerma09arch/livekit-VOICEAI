import { useRoomContext, useTranscriptions } from '@livekit/components-react';

const SEGMENT_ID_ATTR = 'lk.segment_id';

export default function Transcript() {
  const room = useRoomContext();
  const segments = useTranscriptions();

  const localIdentity = room?.localParticipant?.identity;

  if (segments.length === 0) {
    return (
      <aside className="transcript">
        <h2>Transcript</h2>
        <p className="transcript-empty">Say something and it will show up here.</p>
      </aside>
    );
  }

  return (
    <aside className="transcript">
      <h2>Transcript</h2>
      <div className="transcript-scroll">
        {segments.map((seg) => {
          const isUser = seg.participantInfo.identity === localIdentity;
          const key = `${seg.streamInfo.attributes?.[SEGMENT_ID_ATTR] ?? seg.streamInfo.id}:${seg.participantInfo.identity}`;
          return (
            <p key={key} className={isUser ? 'line user' : 'line agent'}>
              <span className="who">{isUser ? 'you' : 'Kilos'}</span>
              <span className="text">{seg.text}</span>
            </p>
          );
        })}
      </div>
    </aside>
  );
}
