# Timestral Uncertainty Arena

Draw a time series with the mouse. [Timestral](https://huggingface.co/collections/mistralai/timestral) forecasts every new step in your browser, online conformal and Bayesian methods calibrate its intervals live, and a ranking compares them as you draw.

![Drawing a series in Timestral Uncertainty Arena](docs/demo.gif)

<details>
<summary>Full dashboard</summary>

![Timestral Uncertainty Arena dashboard: plot, rolling and per-lead coverage and width, ranking](docs/dashboard.png)

</details>

The demo uses the built-in naive baseline. Pick a Timestral checkpoint (5M to 209M) in the settings panel to forecast with the real model, or [Chronos-Bolt](https://huggingface.co/amazon/chronos-bolt-small) (Tiny to Base) to run the same methods on a second foundation model. Both run in the browser.

## Run locally

```bash
npm install
npm run dev        # http://localhost:5173
```

Optional, for the largest checkpoints or a GPU, run the official PyTorch packages locally and choose "Timestral or Chronos-Bolt on a local server" in the app:

```bash
pip install -r server/requirements.txt
python server/serve.py --port 8765
```

## Methods

Rolling split CP, ACI, DtACI, AgACI, quantile tracker, conformal PID, NexCP, ECI, SF-OGD, SAOCP, Bayesian belief CP, BOCPD (+ ACI), changepoint CP and discounted Bayes, all compared with the model's native quantile interval. Chronos-Bolt predicts only the 10–90% quantiles, so its native band beyond 80% extends each tail with a Gaussian shape fitted to that side. References are listed with each method in the settings panel.

The Calibration card under the ranking shows a PIT histogram of the native forecast and a reliability curve (nominal against empirical coverage over several α) for the native quantiles and the focused method, computed online from the same scored pairs.

## Licence

Code: MIT. Timestral weights: Mistral AI, Modified MIT. Chronos-Bolt weights: Amazon, Apache-2.0. The model ports follow the official Apache-2.0 inference code of each.
