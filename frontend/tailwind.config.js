/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        black: "#000000",
        blue: {
          50: "#b4deff",
          100: "#0072CE",
          200: "#004EA8",
          300: "#002F6C",
          400: "#00263E",
          500: "#041A3C",
        },
        grey: {
          100: "#C1C6C8",
          200: "#5B6770",
          300: "#333F48",
        },
        green: {
          100: "#55B838",
          200: "#3fab83",
          300: "#20d18a",
        },
        orange: "#fcae1e",
      },
    },
  },
  plugins: [],
};
