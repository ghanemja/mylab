# Autograder: a guide for teachers

mylab can grade programming homework for you. You write each problem once,
with a few small test scripts, and then:

- students practise in the free mylab web app and press **Check my answers**;
- you grade a whole class on your own computer with one command, or
- you let Gradescope grade every submission automatically.

Nobody needs a MATLAB licence.

**What you need:** Python 3.10 or newer, and mylab:

```bash
pip install mylab-lang
```

This gives you the `mylab-grade` command used below.

Contents: [1. Write an assignment](#1-write-an-assignment) ·
[2. Check it](#2-check-it) · [3. Share it with students](#3-share-it-with-students) ·
[4. Grade on your computer](#4-grade-on-your-computer) ·
[5. Grade on Gradescope](#5-grade-on-gradescope) ·
[Safety](#safety-please-read) · [Questions](#common-questions)

---

## 1. Write an assignment

An assignment is a folder. The quickest start is to let mylab make one:

```bash
mylab-grade init my-assignment
```

A complete example with four problems is in
[`examples/assignment-intro/`](../examples/assignment-intro). It looks like
this:

```
assignment-intro/
  assignment.json            title, introduction, and the list of problems
  problems/
    p1_celsius/
      problem.json           title, points, function name, time limit
      prompt.md              the problem statement students read
      starter/c2f.m          the file students start from
      solution/c2f.m         your reference solution (never given to students)
      tests/
        01_single_values.m   each test script is worth points
        02_vectors.m
        03_matrices.m        marked "hidden": students do not see this one
    p2_ticket_price/ ...
    p3_longest_streak/ ...
    p4_rain_summary/
      data/rain_week.csv     files the students' code reads
      ...
```

### assignment.json

```json
{
  "format": "mylab-assignment-1",
  "title": "Intro Assignment: First Functions",
  "description": "Four short problems to practise writing functions ...",
  "problems": ["p1_celsius", "p2_ticket_price", "p3_longest_streak", "p4_rain_summary"]
}
```

`problems` lists the folder names inside `problems/`, in the order students
see them. Keep `format` exactly as shown. The description may use Markdown.

### problem.json

```json
{
  "title": "Celsius to Fahrenheit",
  "points": 20,
  "function": "c2f",
  "timeout": 10
}
```

- `function` is the name of the function students write. Their file must be
  called `c2f.m`.
- `timeout` is how many seconds each test may take (10 if you leave it out).
  A student's endless loop is stopped after this time.

### prompt.md

The problem statement, in Markdown. Say exactly what the function is called,
what goes in, what comes out, and give one or two examples. Students can
only meet requirements you wrote down.

### starter/ and solution/

`starter/c2f.m` is what students begin with, usually the first line of the
function and a placeholder:

```matlab
function f = c2f(c)
% C2F  Convert degrees Celsius to degrees Fahrenheit.

f = [];   % replace this line with your code

end
```

`solution/c2f.m` is your own answer. It is used to check your tests and it
never leaves your computer.

### tests/

Each file in `tests/` is one test. A test is a short script that calls the
student's function and uses `assert` to state what must be true:

```matlab
% name: Single temperatures
% points: 8
assert(abs(c2f(0) - 32) < 1e-9, 'c2f(0) should be 32 (water freezes)');
assert(abs(c2f(100) - 212) < 1e-9, 'c2f(100) should be 212 (water boils)');
```

**A test passes when the script runs to the end without an error.** The
student then earns all the points of that test; otherwise none. Partial
credit comes from having several tests per problem.

The comment lines at the top are optional settings:

| Line                  | Meaning                                                              |
|-----------------------|----------------------------------------------------------------------|
| `% name: ...`         | The name students see. Without it, the file name is used.            |
| `% points: 8`         | What this test is worth.                                             |
| `% hidden: true`      | Students do not get this test. It is used only when you grade.       |

If you leave out `% points:`, the problem's points are shared equally among
the tests that have none. If every test has `% points:`, they must add up to
the problem's points.

Tips for good tests:

- The second argument of `assert` is the message a student reads when the
  check fails. Make it helpful: `'c2f(0) should be 32'`.
- Compare decimal numbers with a tolerance (`abs(x - 98.6) < 1e-9`), and whole
  numbers, text and sizes with `isequal`.
- Tests run in file-name order, which is why the names start with `01_`, `02_`.
- Give every problem at least one visible test (so students can check their
  work) and at least one hidden test (so the visible answers cannot simply be
  typed in).
- Do not test printed text. Return values and check those.

### data/

Put files that the code reads (for example CSV files) in `data/`. They are
placed next to the student's code when the tests run, and visible ones are
given to students too. Files used only by hidden tests are still sent to
students, so do not put answers in them.

---

## 2. Check it

```bash
mylab-grade check examples/assignment-intro
```

This reads the whole folder and tells you, in plain words, about anything
missing or misspelled. Then it runs **your reference solutions** against
every test (they must pass all of them) and runs **the starter files** as
well. If a starter file already passes a test, you get a warning, because
students would get those points for doing nothing.

```
Intro Assignment: First Functions: 4 problem(s), 100 points
  p1_celsius: 20 points, 3 test(s), 1 hidden
    solution passes  01_single_values.m  (8 points)
    ...
OK: the assignment is ready.
```

Run `check` again every time you change a test.

---

## 3. Share it with students

Build one file that holds everything students need:

```bash
mylab-grade export examples/assignment-intro --out intro.student.json
```

The file contains the problem statements, starter files, data files and the
visible tests. **Hidden tests and your solutions are not in it.**

Give the file to your students (email, your course page, a shared drive).
Then, in the mylab web app:

1. Students open the app in their browser. Nothing needs to be installed.
2. They choose **Assignment** and load the file you gave them (or open a link
   to it that you posted).
3. They write their code and press **Check my answers**. The app runs the
   visible tests and shows which ones pass, and the error message for the
   ones that do not.
4. When they are happy, they hand in their `.m` files the
   way you ask (your course site, Gradescope, email).

Checking in the browser is **practice**. It happens on the student's own
computer, the tests are visible to them, and nothing is sent to you. The
grade comes from the next step.

---

## 4. Grade on your computer

### One student

```bash
mylab-grade run examples/assignment-intro --submission submissions/alice
```

`submissions/alice` is the folder with that student's `.m` files (a `.zip`
file works too). You get a report like this:

```
Intro Assignment: First Functions - alice
========================================================================

Problem 1: Celsius to Fahrenheit (c2f.m)                         20 / 20
  [PASS] Single temperatures                                       8 / 8
  [PASS] A row of temperatures                                     6 / 6
  [PASS] Columns, matrices and fractions (hidden test)             6 / 6

Problem 4: Rain summary (rain_summary.m)                          0 / 30
  [FAIL] Total rainfall                                            0 / 8
         Error: Index exceeds the number of array elements. Index must not exceed 7.
         in rain_summary.m, line 5:  wettest_day = rain(99);
  ...
------------------------------------------------------------------------
Total                                                    20 / 100  (20%)
```

`[PASS]` passed, `[FAIL]` failed (with the error and the line it happened
on), `[TIME]` stopped because it ran too long, `[MISS]` the file was not
handed in.

Students do not have to be tidy. The files may be in one folder, in one
folder per problem, or inside extra folders; the grader looks for the file
named after each function (`c2f.m`, ...). Helper files next to it are used
as well.

Add `--json result.json` to save the full result for your own scripts.

### The whole class

Put every student's work in one folder, one sub-folder or `.zip` per
student. This is how most learning platforms let you download submissions.

```
submissions/
  alice/
  bob.zip
  carol/
```

```bash
mylab-grade batch examples/assignment-intro --submissions submissions \
    --csv grades.csv --reports reports
```

- `grades.csv` has one line per student: the name (the folder or zip name),
  the score for each problem, the total, the maximum and the percentage. It
  opens in any spreadsheet program.
- `reports/` gets a report for every student (`alice.md` and `alice.txt`)
  that you can send back as feedback.
- Add `--jobs 4` to grade four students at the same time on a computer with
  several processor cores.

Reports name hidden tests and show the error, but never show the code of a
hidden test.

---

## 5. Grade on Gradescope

Gradescope can run the same tests each time a student uploads.

1. Build the autograder file:

   ```bash
   mylab-grade gradescope examples/assignment-intro --out autograder.zip
   ```

2. In Gradescope, open your course and choose **Assignments → Create
   Assignment → Programming Assignment**.
3. Set **Autograder points** to the assignment's total (100 for the example).
   Fill in the dates and create the assignment.
4. On the **Configure Autograder** page, upload `autograder.zip` and press
   **Update Autograder**. Gradescope now prepares the grader; this takes a
   few minutes the first time because it installs Python and mylab.
5. Press **Test Autograder** and upload your own reference solutions. You
   should see full marks. Then try a starter file: you should see zero.
6. Publish the assignment. Students upload their `.m` files and see the
   result of each visible test within a minute or so.

Hidden tests appear to students only after you publish grades
(Gradescope's "after published" setting), so they cannot use them to guess
answers before the deadline.

The zip contains your tests and data files but **not** your solutions. If you
change a test, build the zip again and upload it again; Gradescope can then
re-run all submissions.

To see on your own computer exactly what Gradescope would show for one
student:

```bash
mylab-grade run examples/assignment-intro --submission submissions/alice --gradescope results.json
```

---

## Safety: please read

- **Grading runs the students' code on your computer, with your user
  account's permissions.** Code that deletes or reads files could do so. For a
  class you trust, on a normal school computer, this is the same risk as
  opening their files in MATLAB. For anything larger, run `batch` inside a
  container or a virtual machine (for example Docker), or use Gradescope,
  which runs every submission in its own throw-away container.
- Each test runs separately with a time limit, so a crash, an endless loop or
  an `exit` in one student's code only fails that test; it cannot stop the
  grading of the other tests or the other students.
- **Hidden tests and solutions are never put in the student file** made by
  `export`.
- **Tests in the browser are visible to students.** A determined student can
  read them. Treat the web app as practice, and base final grades on hidden
  tests run by you (`run`, `batch`) or by Gradescope.
- While a test runs, the test script sits in the same folder as the student's
  code. Very determined cheating (code that looks for the test file and reads
  it) is possible in principle; if a perfect score looks suspicious, read the
  submitted code.

---

## Common questions

**A student's code works in MATLAB but fails here.** mylab covers the MATLAB
language and the functions used in typical courses, but not everything. Run
your reference solution with `check` first; if it passes, the tests are fine
and the difference is in the student's code. See
[COMPATIBILITY.md](COMPATIBILITY.md), and please report missing functions.

**Can students print things or draw plots?** Yes. Printed text and figures do
not affect the grade. The first lines a failing test printed are shown in the
report, which helps students debug.

**Can a problem need more than one file?** Yes. Students may hand in helper
functions next to the main file, and you can put several files in `starter/`.

**Can I give partial credit by hand?** The grader gives points per test. Open
`grades.csv` in a spreadsheet to adjust scores afterwards.

**What does a student see for a hidden test?** Its name, whether it passed,
and the error message if it failed, but never the test's code. Give hidden
tests a name that says what they check ("Ages exactly on a limit") without
giving the answer away.

**How do I use the grader from Python?**

```python
from mylab.grader import grade_submission

result = grade_submission("examples/assignment-intro", "submissions/alice")
print(result.score, "/", result.max_score)
print(result.report())          # the same text as `mylab-grade run`
data = result.to_dict()         # plain dictionaries and lists, ready for JSON
```
