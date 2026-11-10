# Driving Insights Simulator

Single-page simulator for the four-source Drive Score formula (ESP acceleration, BMS consumption, outside temperature, terrain) with adjustable 60-20-10-10 weights, shown inside the AAOS Vehicle Dynamics screen.

- `index.html` is the whole app (no build step, no dependencies except the Inter font from Google Fonts).
- `validation/` holds the Python reference implementation and the 10 km validation data (test vectors for the Kotlin port).

Deploy on Cloudflare Pages: framework preset "None", build command empty, output directory `/`.
