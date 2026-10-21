# Stitch UI Handoff — Bharat Rozgaar Sentinel

## Objective

Create a more polished, intuitive UI for the existing Bharat Rozgaar Sentinel application. This is an employment-posting AI risk assessment and research project, not a job board. Improve the interface while preserving the working React/FastAPI behavior and the careful, non-definitive risk language.

## Existing implementation to inspect

- Frontend: React 18 + Vite under `frontend/` (`frontend/src/App.jsx`, `frontend/src/style.css`).
- Backend: FastAPI served at `http://127.0.0.1:8000/` in the local demo.
- `GET /health` reports service/model readiness.
- `POST /extract` accepts PDF, PNG, and JPEG files; the UI supports a 15 MB limit and gives users an editable text preview before assessment.
- `POST /predict` receives the four grouped text areas (role and location; company and recruiter; work and qualifications; pay, benefits and next steps) and returns risk level, fraud/genuine probabilities, model, risk indicators, and optional token evidence.
- The current website includes a landing section, posting analyzer, assessment result, how-it-works flow, held-out model comparison, dataset source, project information, and limitations.
- The currently served website defaults to TF-IDF + Logistic Regression. The project also has BiLSTM, hybrid BiLSTM, and DistilBERT evaluation results. Do not imply that all four are used simultaneously for each prediction.
- The existing API routes and response fields are the integration contract. The redesign should not require backend changes.

## Redesign scope

Design the existing web app for desktop and mobile. Keep these user journeys and states:

1. **Landing:** explain the purpose in plain language and offer a clear route to check a posting.
2. **Analyzer:** let users paste information into the existing four plain-language sections or upload a PDF/screenshot. Keep labels concise, and allow missing details.
3. **Upload and extraction:** show empty, selected-file, extracting, text-ready, unreadable/no-text, and error states. Let users edit extracted text before assessment. Retain manual entry when extraction fails.
4. **Assessment:** show risk level, fraud and genuine estimates, active model, indicators, optional wording evidence, and practical independent verification steps. Preserve the user's entry if the request fails.
5. **Research information:** present the measured model comparison and dataset context clearly, with a note that historical held-out results do not guarantee real-world performance.
6. **Service unavailable:** show a calm offline state with a retry/recovery hint; do not discard the form content.

## Visual direction

- Calm, credible, people-first editorial style suitable for a public-interest safety tool.
- Clear typography and hierarchy, generous spacing, responsive cards, strong keyboard focus and readable contrast.
- Warm neutral backgrounds with deep ink/navy and restrained teal/green accents. Use amber/red for risk cues with text/icons as well as color.
- Keep the current Indian employment context subtle and respectful.
- Avoid generic job-board patterns, dense admin dashboards, alarmist scam imagery, excessive gradients, and decorative metrics that suggest unsupported certainty.

## Product and data rules

- Say “AI-based risk assessment”, “risk signals”, and “potentially fraudulent”. Never state that a specific job is proven fraudulent or guaranteed genuine.
- Keep both probabilities and the risk level framed as model estimates, not certainty or legal advice.
- Do not invent accuracy scores, research claims, model capabilities, or guarantees. Use only the existing measured values already displayed by the site.
- Keep the distinction between learned model output and rule-based indicators understandable.
- Do not imply that document OCR itself evaluates fraud; the extracted, user-reviewed text is sent to the existing classifier.
- Do not imply that screenshots, offer letters, or all scam types are represented in the historical training data.

## Ready-to-paste prompt for Stitch

> Inspect the supplied Bharat Rozgaar Sentinel codebase and redesign its existing React web UI. The product is an AI-based risk assessment tool for employment postings in India, not a job board. Improve clarity, trust, accessibility, and mobile usability while preserving the existing FastAPI integration: GET /health, POST /extract for PDF/PNG/JPEG text extraction, and POST /predict for the four grouped text areas and existing assessment response. Design desktop and mobile views for the landing page, four-section analyzer, upload/extraction/editable-preview flow, assessment results, model-comparison/research section, and API-offline/error states. Include empty, loading, success, unreadable-document, and retry states; preserve typed text on errors. The result must show risk level, fraud and genuine probability estimates, the active model, risk indicators, and optional token evidence in a readable hierarchy. Use a calm editorial style, warm neutral background, deep navy/ink text, restrained teal/green, and amber/red only for risk cues; meet accessible contrast and keyboard-focus expectations. Keep the Indian context subtle. Avoid job-board or admin-dashboard styling, alarmist visuals, unsupported charts, invented metrics, and guarantees. Always use “potentially fraudulent” and explain that model output is not proof. Do not change the API contract or claim that the historical dataset validates every offer letter or screenshot. Return implementation-ready screens and components, and flag any idea that needs backend work instead of presenting it as implemented.

## Handoff steps

1. Open [Google Stitch](https://stitch.withgoogle.com/) and sign in.
2. Import the existing codebase. Choose `hitarthshah3/Bharat-Rozgaar-Sentinel` if the import flow offers repository selection and the signed-in Google/Stitch account can access it. The GitHub repository is private, so access may require changing accounts or using an authorized local/codebase import flow.
3. Paste the prompt above. Ask Stitch to create the landing/analyzer/result states at desktop and mobile widths, then refine the chosen design.
4. Check all generated copy and metrics against the project rules above. Export/share the selected design or code and bring it back into the project for integration with the existing components and API.

Google's May 19, 2026 Stitch announcement says Stitch supports bringing existing codebases and design files into its canvas. This session has no Stitch account connector, so the import must be started in Stitch by the project owner.
