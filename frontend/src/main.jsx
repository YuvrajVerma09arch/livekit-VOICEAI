import { createRoot } from 'react-dom/client';
import '@livekit/components-styles';
import './styles.css';
import App from './App.jsx';

createRoot(document.getElementById('root')).render(<App />);