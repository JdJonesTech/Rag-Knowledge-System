# Contributing to JD Jones RAG System

We welcome contributions! Please follow these guidelines to ensure a smooth collaboration process.

## Git Workflow

1. **Main Branch**: The `main` branch is for production-ready code. Do not commit directly to `main`.
2. **Feature Branches**: Always branch off `main` for your work.
   ```bash
   git checkout main
   git pull origin main
   git checkout -b feature/your-feature-name
   ```
3. **Draft PRs**: Feel free to open a Draft PR early if you want feedback.

## Pull Request Guidelines

- Ensure your code follows the existing style.
- Run tests before submitting:
  ```bash
   docker-compose exec api pytest
  ```
- Provide a clear description of what changed and why.
- Link any related issues.

## Environment Variables

Do not commit `.env` files. Ensure any new configuration is added to `.env.example`.

## Coding Standards

- **Python**: Follow PEP 8 guidelines.
- **JavaScript**: Use functional components and hooks for React code.
- **Documentation**: Update README or COLLABORATION docs if your changes affect the setup process.
