import { useCallback, useEffect, useRef, useState } from 'react';
import { useRoomContext, useTranscriptions } from '@livekit/components-react';

const SEGMENT_ID_ATTR = 'lk.segment_id';

// px from the bottom still counts as "at the bottom" (scrollbar + line-height slop)
const STICK_THRESHOLD_PX = 48;

export default function Transcript() {
  const room = useRoomContext();
  const segments = useTranscriptions();

  const localIdentity = room?.localParticipant?.identity;

  const scrollRef = useRef(null);
  const [stickToBottom, setStickToBottom] = useState(true);

  // Track whether the user is at (near) the bottom. Programmatic scrolls land
  // at the bottom too, so they simply confirm the state instead of fighting it.
  const handleScroll = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    setStickToBottom(distanceFromBottom <= STICK_THRESHOLD_PX);
  }, []);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el || !stickToBottom) return;
    el.scrollTop = el.scrollHeight;
  }, [segments, stickToBottom]);

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
      <div ref={scrollRef} className="transcript-scroll" onScroll={handleScroll}>
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
