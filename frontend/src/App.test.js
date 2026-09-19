import { render, screen } from '@testing-library/react';
import App from './App';

test('renders LexiRAG title and branding', () => {
  render(<App />);
  const titleElements = screen.getAllByText(/LexiRAG/i);
  expect(titleElements.length).toBeGreaterThan(0);
});
