import { VoiceAssistantControlBar } from '@livekit/components-react';

export default function ControlBar() {
  return (
    <footer className="control-bar">
      <VoiceAssistantControlBar
        controls={{ microphone: true, leave: true }}
        saveUserChoices={false}
      />
    </footer>
  );
}
