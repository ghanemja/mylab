# Releasing mylab

This is a checklist for the owner of the repository. You do not need to know
about packaging. Follow the steps in order.

There are two parts:

- **Part A: one-time setup.** You do this once. Only steps 1 and 2 are needed
  for a first release. The rest can wait, or be skipped for good.
- **Part B: each release.** About 10 minutes of your time, then about 10
  minutes of waiting.

Some names you will see:

| Name | What it is |
|---|---|
| `mylab-lang` | The name of the package on PyPI and conda-forge. (`mylab` was taken.) |
| `mylab` | The command people type, and the Python module. |
| `ghanemja/mylab` | The GitHub repository. If yours has another name, use that name wherever this page says `ghanemja/mylab`. |

## Part A: one-time setup

| Step | What you get | Your time | Can you skip it? |
|---|---|---|---|
| 1. Make the repository public | People can see and install mylab | 2 min | No |
| 2. PyPI | `pip install mylab-lang` works | 10 min | No, this is the main way to install |
| 3. GitHub Pages | The web app is online | 3 min | Yes |
| 4. VS Code Marketplace | The extension can be installed from inside VS Code | 20 min | Yes. The `.vsix` file on each release works without it |
| 5. Open VSX | The same, for VSCodium, Cursor and similar editors | 15 min | Yes |
| 6. conda-forge | `conda install -c conda-forge mylab-lang` works | 30 min, then days of waiting | Yes |
| 7. Datasets on Hugging Face and Zenodo | The datasets are easy to find and to cite | 20 min each | Yes |

Most steps end with a **variable** or a **secret** in GitHub. Both are made in
the same place:

> Repository page → **Settings** → **Secrets and variables** → **Actions**.
> Use the **Variables** tab for variables and the **Secrets** tab for secrets.
> Select **New repository variable** or **New repository secret**, type the
> name and the value, and save.

Names must be typed exactly as shown, in capitals.

### 1. Make the repository public (2 minutes)

1. Open the repository on GitHub and go to **Settings**.
2. Scroll to the bottom, to **Danger Zone**.
3. Select **Change visibility** → **Change to public**, and confirm.

Before you do, read the commit history once for anything private (passwords,
personal files). Making a repository public shows its whole history.

### 2. PyPI (10 minutes)

PyPI is where `pip install` gets packages. You will not create a password or
token for GitHub. Instead you tell PyPI to trust this repository's release
workflow. This is called *trusted publishing*.

1. Create an account at <https://pypi.org/account/register/>. Confirm your
   email address, and turn on two-factor authentication when asked. PyPI
   requires it.
2. Go to <https://pypi.org/manage/account/publishing/>.
3. Under **Add a new pending publisher**, choose the **GitHub** tab and fill in:

   | Field | Type this |
   |---|---|
   | PyPI Project Name | `mylab-lang` |
   | Owner | `ghanemja` |
   | Repository name | `mylab` |
   | Workflow name | `release.yml` |
   | Environment name | `pypi` |

4. Select **Add**.
5. In GitHub, create the repository **variable** `PUBLISH_TO_PYPI` with the
   value `true`.

That is all. The project page on PyPI appears with the first release.

If you leave out item 5, releases still work. They are just not sent to PyPI.

Optional: in GitHub, **Settings** → **Environments** → **pypi**, you can add
yourself as a required reviewer. Each upload to PyPI then waits for you to
approve it. (The environment appears after the first release, or you can
create it yourself with **New environment** and the name `pypi`.)

### 3. GitHub Pages: the web app (3 minutes, optional)

1. Repository → **Settings** → **Pages**.
2. Under **Build and deployment**, set **Source** to **GitHub Actions**.
3. Create the repository **variable** `DEPLOY_PAGES` with the value `true`.
4. Go to the **Actions** tab, choose **Deploy web app to GitHub Pages** in the
   list on the left, and select **Run workflow** → **Run workflow**.

After about 2 minutes the app is at `https://ghanemja.github.io/mylab/`. From
now on it is updated on every push to `main`.

### 4. VS Code Marketplace (20 minutes, optional)

This lets people find "mylab" in the Extensions view of VS Code. Without it,
they can still install the `.vsix` file from the release page
([how](VSCODE.md#2-install-the-extension)).

You need a Microsoft account. The steps are on Microsoft's sites and their
pages change from time to time. If a button has moved, see
<https://code.visualstudio.com/api/working-with-extensions/publishing-extension>.

1. **Create a publisher.** Go to <https://marketplace.visualstudio.com/manage>,
   sign in, and select **Create publisher**. Choose an **ID** (letters, digits
   and hyphens, for example `ghanemja`). You cannot change the ID later.
2. **Tell the extension your publisher ID.** The extension currently says
   `"publisher": "mylab"`, which is a placeholder. Do one of these:
   - Create the repository **variable** `VSCODE_PUBLISHER` with your ID as the
     value. (Easiest. The release workflow puts it in for you.)
   - Or edit `editors/vscode/package.json`, change the `publisher` line to
     your ID, and commit.
3. **Create a token.**
   1. Go to <https://dev.azure.com>. Sign in with the same Microsoft account.
      If it asks you to create an organization, accept the suggested name.
   2. Select the **User settings** icon at the top right (a person with a
      cog) → **Personal access tokens** → **New Token**.
   3. Fill in: **Name**: `vsce`. **Organization**: **All accessible
      organizations**. **Expiration**: the longest offered (one year).
      **Scopes**: **Custom defined**, then **Show all scopes**, find
      **Marketplace** and tick **Manage**.
   4. Select **Create** and copy the token. It is shown only once.
4. In GitHub, create the repository **secret** `VSCE_PAT` and paste the token
   as its value.

The token expires. Put a reminder in your calendar to repeat items 3 and 4
before then. If it has expired, a release still succeeds on GitHub and PyPI;
only the Marketplace job fails, and you can run it again after replacing the
secret (**Actions** → the failed run → **Re-run failed jobs**).

### 5. Open VSX (15 minutes, optional)

Open VSX is the extension gallery used by VSCodium, Cursor, Gitpod and other
editors that cannot use Microsoft's Marketplace.

1. Go to <https://open-vsx.org> and sign in with GitHub.
2. Open your profile (top right) → **Settings**. Follow the prompt to sign the
   **Publisher Agreement**. It asks you to create or connect an Eclipse
   account. Use the same email address as on GitHub.
3. In **Settings** → **Access Tokens**, select **Generate New Token**, name
   it `mylab`, and copy the token.
4. Create your namespace. It must be the same as your publisher ID from
   step 4 (or `mylab` if you kept the placeholder). In a terminal on your
   computer (needs Node.js):

   ```bash
   npx ovsx create-namespace YOUR-PUBLISHER-ID --pat PASTE-THE-TOKEN-HERE
   ```

5. In GitHub, create the repository **secret** `OVSX_PAT` and paste the token.

If you skipped step 4 (Marketplace), still do item 2 of step 4 so that the
extension carries your own ID.

### 6. conda-forge (optional, about 30 minutes, then waiting)

conda-forge is the community channel for conda. Volunteers review each new
package, so the first submission takes a few days to a few weeks. After that,
a robot updates the conda package whenever a new version appears on PyPI. You
only need to press "merge".

Do this **after** the first release is on PyPI (Part B), because the recipe
downloads the package from there.

1. On the GitHub release page of your new version, download
   **`conda-meta.yaml`**. It is the recipe from `packaging/conda/meta.yaml`
   with the version and checksum already filled in.
2. Go to <https://github.com/conda-forge/staged-recipes> and select **Fork**.
3. In your fork, select **Add file** → **Create new file**. As the file name,
   type `recipes/mylab-lang/meta.yaml`. (Typing the `/` creates the folders.)
4. Paste the content of `conda-meta.yaml`. Near the bottom, under
   `recipe-maintainers`, make sure your own GitHub user name is listed.
5. Select **Commit changes**, choose **Create a new branch**, then
   **Propose changes** and **Create pull request**. Read the checklist in the
   pull request text and tick its boxes.
6. Wait for the automatic checks (about 20 minutes). If one fails, open its
   details. The message usually names the line to fix.
7. When the checks are green, add a comment with the text the pull request
   template gives for asking for a review (for a Python package:
   `@conda-forge/help-python, ready for review!`).
8. A reviewer merges it, or asks for changes. After the merge, a repository
   named `conda-forge/mylab-lang-feedstock` is created and you are its
   maintainer. About an hour later, `conda install -c conda-forge mylab-lang`
   works.

For later releases you do nothing by hand: a robot opens a pull request in the
feedstock, and you merge it when its checks are green. If you add or remove a
dependency or a command in `pyproject.toml`, make the same change in the
feedstock's `recipe/meta.yaml` (and in `packaging/conda/meta.yaml` here).

### 7. Datasets on Hugging Face and Zenodo (optional, about 20 minutes each)

The project has two datasets (see [datasets/README.md](../datasets/README.md)).
Hugging Face makes them easy to load in code. Zenodo gives them a DOI, a
permanent identifier that papers can cite.

**Build the files.** In a terminal, in the repository folder:

```bash
# MATLAB Tasks: a few seconds
python datasets/matlab-tasks/tools/export_jsonl.py --full
```

This writes `datasets/matlab-tasks/dist/matlab_tasks.jsonl`, with prompts,
reference solutions and tests. You can skip this command: every GitHub release
already has the file attached as `matlab_tasks_full.jsonl`, next to
`matlab_tasks.jsonl`, which has the prompts only.

```bash
# MATLAB Corpus: downloads about 400 MB, takes 10 to 30 minutes
python datasets/matlab-corpus/tools/fetch.py && python datasets/matlab-corpus/tools/package.py
```

This writes `datasets/matlab-corpus/dist/matlab-corpus.tar.gz`. The archive
contains other people's code, each under its own licence, with a `NOTICE.md`
that lists every source. Read the licence section of
[datasets/matlab-corpus/README.md](../datasets/matlab-corpus/README.md) before
you upload it anywhere, and copy that section into the dataset's description.

**Hugging Face** (once per dataset):

1. Create an account at <https://huggingface.co/join>.
2. Go to <https://huggingface.co/new-dataset>. Name: `matlab-tasks`. Licence:
   `mit`. Visibility: public. Select **Create dataset**.
3. On the dataset page, open **Files and versions** → **Add file** →
   **Upload files**. Drag in the `.jsonl` files and select **Commit changes**.
4. Select **Create dataset card** (or edit `README.md`) and paste the text of
   `datasets/matlab-tasks/README.md`.
5. For the corpus, repeat with the name `matlab-corpus`, the file
   `matlab-corpus.tar.gz`, licence `other`, and the text of
   `datasets/matlab-corpus/README.md`.

To update a dataset later, upload the new files the same way. They replace the
old ones.

**Zenodo:**

1. Sign in at <https://zenodo.org> with GitHub.
2. Select **New upload**. Drag in the files of one dataset.
3. Fill in: **Resource type**: Dataset. **Title**: for example
   `MATLAB Tasks: programming tasks with reference solutions and tests`.
   **Creators**: you. **Description**: paste the first paragraphs of the
   dataset's README. **Licence**: MIT for the tasks; for the corpus choose
   **Other (Open)** and explain in the description that each repository keeps
   its own licence. **Version**: the mylab version.
4. Select **Publish**. This cannot be undone, so check the files first.
5. Zenodo shows a DOI badge. Paste it into the dataset's README.

For a new version later, open the record and select **New version**.

A shortcut for the code itself: in Zenodo, open your account menu → **GitHub**
and switch on `ghanemja/mylab`. Every GitHub release is then archived on Zenodo
with a DOI, without any further action.

## Part B: each release

A release is made by pushing a **tag**, a name for one commit, such as
`v1.0.1`. GitHub then does the rest.

### Choose the version number

Versions look like `MAJOR.MINOR.PATCH`, for example `1.2.0`.

- Bug fixes only: raise the last number (`1.0.0` → `1.0.1`).
- New functions or features: raise the middle number (`1.0.1` → `1.1.0`).
- Changes that break existing programs: raise the first number.

### Steps

Below, replace `X.Y.Z` with your new version number.

1. **Set the version in two files.** They must be the same.
   - `pyproject.toml`: the line `version = "X.Y.Z"`
   - `mylab/__init__.py`: the line `__version__ = "X.Y.Z"`

   (The VS Code extension takes its version from the tag automatically.)
2. **Update `CHANGELOG.md`.** Add a heading `## X.Y.Z` at the top and list
   what changed, in a few bullet points. If the extension changed, do the same
   in `editors/vscode/CHANGELOG.md`.
3. **Check on your computer** (2 minutes):

   ```bash
   python scripts/gen_editor_data.py     # refreshes the editor's function list
   pytest -q
   ruff check .
   ```

4. **Commit and push:**

   ```bash
   git add -A
   git commit -m "Release X.Y.Z"
   git push origin main
   ```

5. **Run the checks on your own machine:** `scripts/check.sh --web`. Fix
   anything that fails first. (The test workflows on GitHub no longer run on
   every push. You can still start them by hand from the **Actions** tab to
   test on Windows and macOS.)
6. **Tag and push the tag:**

   ```bash
   git tag vX.Y.Z && git push origin vX.Y.Z
   ```

   Note the `v` in front of the number.

### What happens next (about 10 minutes, automatic)

The **Release** workflow (`.github/workflows/release.yml`) runs. You can watch
it in the **Actions** tab. It:

1. Checks that the tag matches the version in the two files, and runs the
   tests. If anything fails, it stops here and nothing is published.
2. Builds the Python package (a `.whl` and a `.tar.gz` file).
3. Builds the VS Code extension (`mylab-X.Y.Z.vsix`).
4. Exports the MATLAB Tasks dataset (two `.jsonl` files).
5. Creates the GitHub release with all of these files, plus
   `conda-meta.yaml`, and writes release notes from the commit messages.
6. Publishes to PyPI, if `PUBLISH_TO_PYPI` is `true`.
7. Publishes the extension to the VS Code Marketplace, if the secret
   `VSCE_PAT` exists, and to Open VSX, if `OVSX_PAT` exists.

Steps that are not set up are skipped. They show as grey, not red. That is fine.

### Check that it worked (5 minutes)

1. **GitHub:** on the repository page, select **Releases**. The new release
   should list six files: one `.whl`, one `.tar.gz`, one `.vsix`,
   `matlab_tasks.jsonl`, `matlab_tasks_full.jsonl` and `conda-meta.yaml`.
   (GitHub adds two "Source code" archives by itself.)
2. **PyPI:** open <https://pypi.org/project/mylab-lang/>. It should show the
   new version. Then try it in a fresh environment:

   ```bash
   python -m venv /tmp/try-mylab
   /tmp/try-mylab/bin/pip install mylab-lang==X.Y.Z
   /tmp/try-mylab/bin/mylab --version
   ```

   (On Windows, use a folder such as `C:\Temp\try-mylab`, and `Scripts`
   instead of `bin`.)
3. **VS Code Marketplace** (if set up): search for "mylab" in the Extensions
   view. A new version can take up to 15 minutes to appear.
4. **Edit the release notes** if you like: on the release page, select the
   pencil icon. Pasting the new section of `CHANGELOG.md` at the top is enough.

### If something goes wrong

- **The workflow failed before anything was published** (the test or build
  jobs are red). Fix the problem, commit and push, then move the tag:

  ```bash
  git tag -d vX.Y.Z
  git push origin :refs/tags/vX.Y.Z
  git tag vX.Y.Z && git push origin vX.Y.Z
  ```

- **"The three versions differ".** The tag and the two version lines do not
  match. Correct the files (step 1), commit and push, then move the tag as
  above.
- **One publishing job failed** (PyPI, Marketplace or Open VSX) but the rest
  worked. Open the failed run in the **Actions** tab, read the message, fix
  the setup (most often an expired token), then select **Re-run failed jobs**.
- **A broken version is already on PyPI.** PyPI never allows the same version
  number to be uploaded twice. Fix the problem and release the next number
  (for example `1.0.2`). You can hide the broken one on PyPI: project page →
  **Manage** → **Releases** → **Options** → **Yank**.

### Try the workflow without publishing

In the **Actions** tab, choose **Release** on the left, then **Run workflow**.
This runs the tests and builds every file, but creates no release and
publishes nothing. The built files can be downloaded at the bottom of the
run's page, under **Artifacts**. Use this to test changes to the release
process, or to get a `.vsix` to try by hand.
