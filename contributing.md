# Contributing to Ethical AI Toolkit

Thank you for your interest in contributing to the **Ethical AI Toolkit**! We welcome contributions of all kinds, whether they are bug fixes, feature requests, documentation improvements, or entirely new modules. Please follow the guidelines below to ensure a smooth collaboration.

---

## Table of Contents
- [How to Contribute](#how-to-contribute)
- [Setting Up Your Environment](#setting-up-your-environment)
- [Making Changes](#making-changes)
- [Submitting a Pull Request](#submitting-a-pull-request)
- [Code of Conduct](#code-of-conduct)
- [Contact](#contact)

---

## How to Contribute

1. **Fork the Repository**: First, fork this repository to create your own copy.
2. **Clone Your Fork**: Clone your forked repository to your local machine.
    ```bash
    git clone https://github.com/YOUR_USERNAME/ethical_AI_toolkit.git
    cd ethical_AI_toolkit
    ```
3. **Create a New Branch**: Always create a new branch for your changes to keep your work organized and separate from the main branch.
    ```bash
    git checkout -b feature/new-feature
    ```

---

## Setting Up Your Environment

To set up a development environment for the Ethical AI Toolkit, follow these steps:

1. **Install Dependencies**:
    - Ensure you have Python 3.7 or newer.
    - Install dependencies from `requirements.txt`:
      ```bash
      pip install -r requirements.txt
      ```
2. **Additional Setup**:
    - If you are working on specific features that require APIs (e.g., GDELT, Newscatcher), please obtain API keys and set them as environment variables or configure them within your development environment as described in the README.
3. **Verify Setup**:
    - Run the Flask application to confirm everything is set up:
      ```bash
      flask run
      ```

---

## Making Changes

1. **Add New Features or Fix Issues**:
    - Follow best coding practices, and write clear, concise code.
    - **Document your changes**: If you add a new feature, update relevant documentation (e.g., `README.md`) and add comments in the code for clarity.
    - **Testing**: If possible, test your changes locally to verify they work as expected.

2. **Update `CONTRIBUTE.md` if Necessary**:
    - If your changes require additional setup or specific guidelines, please update this `CONTRIBUTE.md` file to reflect that.

---

## Submitting a Pull Request

Once you have made your changes and committed them, follow these steps to submit a pull request:

1. **Push Your Changes**:
    ```bash
    git add .
    git commit -m "Add description of your changes"
    git push origin feature/new-feature
    ```

2. **Open a Pull Request**:
    - Go to the original repository on GitHub and click on **Compare & pull request**.
    - Ensure the base repository points to the main branch of the Ethical AI Toolkit and the head repository points to your feature branch.
    - Provide a detailed description of your changes, including the purpose of the pull request and any relevant details.

3. **Respond to Review Comments**:
    - Maintain an open line of communication. Be prepared to respond to any review comments or requests for changes.

---

## Code of Conduct

All contributors must adhere to the following principles:

1. **Respectful Communication**: Be considerate and respectful in all communications. 
2. **Follow the Guidelines**: Adhere to the contribution guidelines to ensure smooth collaboration.
3. **Stay on Topic**: Keep discussions relevant to the project and avoid unrelated topics.

For more information, please review our [Code of Conduct](CODE_OF_CONDUCT.md).

---

## Contact

If you have any questions, suggestions, or need further assistance, feel free to contact the maintainers through GitHub Issues or Discussions.

Thank you for your interest in making the Ethical AI Toolkit better!
