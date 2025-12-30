# Persistent prompt

From now on, I want to persist the prompt in `viewer.js` so that every time the user generates a quadrant with a prompt, that prompt is saved + reapplied to subsequent generations. Let's add an indicator to either the toast or the button to indicate that a prompt is saved and being applied.

## Implementation (completed)

- **localStorage persistence**: Prompt is saved to `viewer_saved_prompt` key when user generates with a prompt
- **Auto-apply**: Saved prompt is automatically applied to all subsequent generations (via "Generate" button)
- **Visual indicator**: The "+ Prompt" button turns green with a pulsing dot indicator when a prompt is saved
- **Button tooltip**: Shows the saved prompt text on hover
- **Dialog pre-fill**: Opening the prompt dialog shows the saved prompt pre-filled
- **Clear functionality**: "Clear Saved" button in the dialog to remove the saved prompt
- **Toast notifications**: Shows confirmation when prompt is saved or cleared
