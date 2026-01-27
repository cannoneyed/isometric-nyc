import ReactDOM from "react-dom/client";
import App from "./App";
import "./styles/global.css";

// StrictMode disabled to prevent double tile loading in development
ReactDOM.createRoot(document.getElementById("root")!).render(<App />);
