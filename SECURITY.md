# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in LiveSTT, please report it by:

1. **DO NOT** open a public issue
2. Email the maintainers directly (or use GitHub Security Advisories)
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

We will respond within 48 hours and work on a fix as quickly as possible.

## Security Best Practices

When using LiveSTT:

- **Never commit API keys** to the repository
- Use environment variables for sensitive configuration
- Keep dependencies up to date (`pip install --upgrade -r requirements.txt`)
- Run the server behind a reverse proxy (nginx, Apache) in production
- Use HTTPS in production environments
- Limit file upload sizes to prevent DoS attacks
- Regularly update the Whisper model and dependencies

## Known Security Considerations

- File uploads are limited by `MAX_UPLOAD_SIZE_MB` in config
- Path traversal protection is implemented for file operations
- Filename suffixes are sanitized to prevent injection
- Temporary files are automatically cleaned up

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| Latest  | ✅ Yes             |
| Older   | ❌ No              |

We recommend always using the latest version from the main branch.
