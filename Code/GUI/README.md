# Radian - Sports Analytics Dashboard

This is a sports analytics dashboard built with Next.js, TypeScript, Shadcn UI, and Supabase. The application allows for real-time tracking and visualization of player movement data collected from ESP32 devices via the Web Serial API.

## Features

- Connect to ESP32 devices using Web Serial API
- Record and visualize player movement data in real-time
- Store and analyze historical session data
- Manage player profiles and session information
- Dark mode support with customizable theme

## Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

## Project Structure

- `app/`: Next.js App Router pages and layouts
- `components/`: Reusable UI components
- `lib/`: Utility functions and services
  - `supabase/`: Supabase client and database operations
  - `serial/`: Web Serial API utilities
  - `data/`: Data processing and transformation
- `hooks/`: Custom React hooks
- `types/`: TypeScript type definitions
- `providers/`: Context providers for state management

## Technologies Used

- [Next.js](https://nextjs.org/) - React framework
- [TypeScript](https://www.typescriptlang.org/) - Type safety
- [Shadcn UI](https://ui.shadcn.com/) - Component library
- [Tailwind CSS](https://tailwindcss.com/) - Styling
- [Supabase](https://supabase.com/) - Backend and database
- [Web Serial API](https://developer.mozilla.org/en-US/docs/Web/API/Serial) - Device communication

## Development

This project follows a structured implementation plan with clearly defined steps. See the `project-steps.md` file for details on the implementation roadmap.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.
