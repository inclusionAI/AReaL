# Beginner's Guide to Contributing

Welcome! This guide will walk you through making your first contribution to AReaL, even
if you've never contributed to open source before.

## Prerequisites

- Basic Python knowledge
- Git installed on your computer
- GitHub account
- Text editor or IDE (VS Code, PyCharm, etc.)

## Step 1: Set Up Your Environment

### 1.1 Fork the Repository

1. Go to [https://github.com/inclusionAI/AReaL](https://github.com/inclusionAI/AReaL)
1. Click the "Fork" button in the top right
1. This creates your own copy of the repository

### 1.2 Clone Your Fork

```bash
# Replace YOUR-USERNAME with your GitHub username
git clone https://github.com/YOUR-USERNAME/AReaL
cd AReaL
```

### 1.3 Add Upstream Remote

This lets you sync with the main repository:

```bash
git remote add upstream https://github.com/inclusionAI/AReaL.git
git remote -v  # Verify it was added
```

### 1.4 Install AReaL

```bash
# Install in development mode
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

### 1.5 Set Up Pre-commit Hooks

This automatically formats your code:

```bash
pip install pre-commit
pre-commit install
```

## Step 2: Find Something to Work On

### Option A: Good First Issues

Browse issues labeled
[`good first issue`](https://github.com/inclusionAI/AReaL/labels/good%20first%20issue).
These are beginner-friendly tasks like:

- Fixing typos in documentation
- Adding code comments
- Improving error messages
- Writing tests for existing code

### Option B: Documentation

Documentation improvements are a great way to start:

- Fix typos or grammar
- Clarify confusing explanations
- Add examples
- Improve README files

### Option C: Report a Bug

If you found a bug, create an issue first before fixing it.

## Step 3: Create a Branch

Always create a new branch for your changes:

```bash
# Sync with upstream first
git fetch upstream
git checkout main
git merge upstream/main

# Create your branch
git checkout -b fix/typo-in-readme
# or
git checkout -b docs/improve-quickstart
```

**Branch naming:**

- `fix/description` - Bug fixes
- `docs/description` - Documentation
- `feature/description` - New features

## Step 4: Make Your Changes

### Example: Fix a Typo in Documentation

1. Open the file (e.g., `docs/tutorial/quickstart.md`)
1. Make your change
1. Save the file

### Example: Add Code Comments

1. Find a function that needs comments
1. Add helpful comments explaining what it does
1. Save the file

## Step 5: Test Your Changes

### For Documentation Changes

```bash
# Install jupyter-book
pip install jupyter-book

# Build docs
jb build docs

# Open docs/_build/html/index.html in your browser to preview
```

### For Code Changes

```bash
# Run relevant tests
pytest -s -v areal/tests/test_<related>.py

# Or run all tests (takes a while)
pytest -s -v areal/tests/
```

## Step 6: Commit Your Changes

### Stage Your Changes

```bash
# See what you changed
git status

# Add specific files
git add docs/tutorial/quickstart.md

# Or add all changes
git add .
```

### Commit With a Good Message

```bash
# Use conventional commit format
git commit -m "docs: fix typo in quickstart guide"
```

**Commit message format:**

- `docs:` - Documentation changes
- `fix:` - Bug fixes
- `feat:` - New features
- `test:` - Test additions

**The pre-commit hooks will run automatically** and format your code. If they make
changes, you'll need to add and commit again:

```bash
git add .
git commit -m "docs: fix typo in quickstart guide"
```

## Step 7: Push to Your Fork

```bash
git push origin fix/typo-in-readme
```

If this is your first push, you might need to set the upstream:

```bash
git push --set-upstream origin fix/typo-in-readme
```

## Step 8: Create a Pull Request

1. Go to your fork on GitHub (https://github.com/YOUR-USERNAME/AReaL)
1. You'll see a banner saying "Compare & pull request" - click it
1. Fill out the PR template:
   - **Title:** Brief description (e.g., "docs: fix typo in quickstart guide")
   - **Description:** What you changed and why
   - **Related Issue:** Link to the issue if there is one
   - Check the boxes in the checklist
1. Click "Create pull request"

## Step 9: Respond to Feedback

A maintainer will review your PR and may:

- Approve it immediately (🎉)
- Request changes
- Ask questions

**If changes are requested:**

1. Make the changes locally
1. Commit them:
   ```bash
   git add .
   git commit -m "address review feedback"
   ```
1. Push to your branch:
   ```bash
   git push origin fix/typo-in-readme
   ```

The PR will automatically update!

## Step 10: Celebrate! 🎉

Once your PR is merged:

- Your contribution is now part of AReaL!
- You'll appear in the contributors list
- You can move on to bigger contributions

## Common Issues and Solutions

### "Pre-commit hooks failed"

This means the formatting tools made changes. Just add and commit again:

```bash
git add .
git commit -m "your message"
```

### "CI checks are failing"

Click on the failing check to see what went wrong:

- **Formatting:** Run `pre-commit run --all-files` locally
- **Tests:** Run `pytest -s -v areal/tests/` to see which tests fail
- Ask for help in the PR if you're stuck!

### "Conflicts with main branch"

Your branch is out of sync. Update it:

```bash
git fetch upstream
git merge upstream/main
# Resolve any conflicts
git add .
git commit -m "merge upstream changes"
git push origin your-branch-name
```

### "I made a mistake in my commit"

You can fix the last commit:

```bash
# Make your changes
git add .
git commit --amend --no-edit  # Keeps same message
# or
git commit --amend -m "new message"  # Changes message
git push --force origin your-branch-name  # Force push to update PR
```

## Getting Help

**Stuck? Don't worry!** Everyone gets stuck sometimes.

- **Ask in your PR:** Leave a comment asking for help
- **GitHub Discussions:**
  [Post a question](https://github.com/inclusionAI/AReaL/discussions)
- **WeChat Group:** [Join our community](../../assets/wechat_qrcode.png)
- **Issue Comments:** Ask on the related issue

## Next Steps

After your first contribution:

1. **Try more good first issues:** Build confidence with similar tasks
1. **Explore the codebase:** Read code to understand how things work
1. **Tackle medium difficulty issues:** Graduate to more challenging tasks
1. **Help others:** Answer questions in Discussions
1. **Propose features:** Share your ideas for improvements

## Resources

- [Contributing Guide](../../CONTRIBUTING.md) - Full contribution guidelines
- [Development Setup](development-setup.md) - Detailed environment setup
- [Testing Guide](testing-guide.md) - How to write and run tests
- [Code Style](code-style.md) - Coding standards and best practices

______________________________________________________________________

**Remember:** Every expert was once a beginner. Don't be afraid to ask questions and
make mistakes. We're here to help! 🙂
