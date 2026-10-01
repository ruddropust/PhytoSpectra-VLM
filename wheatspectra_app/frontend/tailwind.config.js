/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        wheat: {
          50: '#fbf9f1',
          100: '#f5f0de',
          200: '#ebe0bc',
          300: '#dc logic',
          400: '#d1b96a',
          500: '#b89c44',
          600: '#9b7e34',
          700: '#7c622b',
          800: '#665028',
          900: '#564325',
        },
        phyto: {
          50: '#f0fdf4',
          100: '#dcfce7',
          200: '#bbf7d0',
          300: '#86efac',
          400: '#4ade80',
          500: '#22c55e',
          600: '#16a34a',
          700: '#15803d',
          800: '#166534',
          900: '#14532d',
        },
      },
    },
  },
  plugins: [],
}
