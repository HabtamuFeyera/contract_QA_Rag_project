import { render, screen } from '@testing-library/react';
import App from './App';

test('renders LexiRAG title and branding', () => {
  render(<App />);
  const titleElement = screen.getByText(/LexiRAG/i);
  expect(titleElement).toBeInTheDocument();
});
