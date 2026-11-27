# MathWorks licence terms and validating mylab against MATLAB

**This is not legal advice.** It was written by an AI assistant from the
documents listed below. It reports what those documents say and offers a
reading of them; it is no substitute for a lawyer who can see the current
terms and the facts. Treat every interpretation here as a question to put to
counsel, not an answer.

- Review date: 26 November 2026.
- Status: **MATLAB validation is on hold.** Nothing was run on MATLAB, MATLAB
  Online or a MathWorks CI licence for this review, and no MATLAB workflow
  was set up.
- Update, 26 November 2026: the maintainer has confirmed that they hold no
  MathWorks licence and that nobody has ever run the kit on MATLAB. The
  kit's MATLAB Online instructions have been withdrawn, along with the
  related text in the README, COMPATIBILITY, CLEANROOM and CHANGELOG (steps
  1–3 in section 5). Line numbers in section 4 refer to the repository as
  reviewed, at commit `4d2f4dd`.
- Scope: the three routes for running mylab's validation kit
  ([validation/](../validation/)) on real MATLAB: MATLAB Online's free basic
  tier, the free MATLAB licence in GitHub Actions, and (for comparison) any
  other MathWorks licence.

## Contents

1. [Short answer](#1-short-answer)
2. [What was fetched](#2-what-was-fetched)
3. [Clauses, document by document](#3-clauses-document-by-document)
4. [The four questions](#4-the-four-questions)
5. [Recommendation](#5-recommendation)
6. [What could not be fetched or verified](#6-what-could-not-be-fetched-or-verified)

---

## 1. Short answer

| Route | Rating | One-line reason |
|---|---|---|
| Run the kit on **MATLAB Online (free basic tier)** | **Unsafe** | The Online Services Agreement §3.2(3) forbids using or testing a Service "in order to make" a product whose principal purpose is "the same or similar functions". That describes the kit's purpose. |
| Run the kit in **GitHub Actions with the free public-project licence** | **Unsafe** (terms unpublished) | MathWorks publishes no terms for this licence; they ship inside MATLAB. Every Software License Agreement text I could read has the same clause (§3.4), plus "Licensee shall not use a Program to compete with the products or businesses of MathWorks". |
| Any other MathWorks licence (Home, Student, academic, trial, batch token) | **Unsafe** | Same §3.4. Home and Student licences also bar academic or professional use, and the batch-token pilot is closed to new requests. |
| **Documented behaviour + GNU Octave** (what mylab does today) | **Safe** for people who don't hold a MathWorks licence; **uncertain** for those who do | The Online Services Agreement does not cover the public documentation. Octave's licence (GPLv3) places no conditions on running it. The SLA's "Program" includes "Documentation", so licensees are in a greyer position. |
| **User-submitted difference reports** | **Uncertain** | The risk moves to the reporter, whose own MathWorks licence may forbid it. mylab's maintainers are not parties to it. One-off reports are much lower risk than inviting people to run a test campaign. |
| **Ask MathWorks for written permission** | **Safe** to ask | It is the only route that can make MATLAB validation clearly allowed. |

When this review was written, the repository told people to use the unsafe
route: `validation/README.md` lines 29–31 sent users to MATLAB Online with a
free account, and the kit stopped itself at 12 minutes to stay under the
basic tier's 15-minute compute limit. Nobody followed those instructions,
and they have since been withdrawn (section 5).

---

## 2. What was fetched

| Document | Where | Version / date | How it was read | Status |
|---|---|---|---|---|
| MathWorks Software License Agreement (SLA), **current** | `mathworks.com/license` (now **404**). MathWorks says it is shipped only as `license_agreement.txt` inside an installation ([Answers 1979884](https://www.mathworks.com/matlabcentral/answers/1979884-where-can-i-find-the-mathworks-software-license-agreement), updated 21 Nov 2025) | Unknown | Not read | **Not obtained.** Reading it needs MATLAB or MATLAB Online, which this review did not use. |
| SLA, **03/2017 text**, from the R2019b installer | Third-party mirror: [mib.helsinki.fi/Matlab_license.txt](http://mib.helsinki.fi/Matlab_license.txt) | SLA "03/2017"; followed by the R2019b Program Offering Guide, "09/2019" | Downloaded and read in full | Historical. Quoted below. |
| SLA, **03/2017 text**, R2020a | Third-party mirror: [zim.uni-wuppertal.de/…/MathWorks_SLA_R2020a.pdf](https://zim.uni-wuppertal.de/fileadmin/zim/dienste/pdf/MathWorks_SLA_R2020a.pdf) | "03/2017" | Downloaded. All 14 clauses quoted below were checked word for word against it. | Historical. Identical in the clauses quoted. |
| SLA, R2014a era | Third-party mirror: [webpages.tuni.fi/…/MATLAB_license.txt](https://webpages.tuni.fi/cytospectre/cytospectre/MATLAB_license.txt) | References "thirdpartylicenseagreementsR2014a" | Downloaded | Used only to show that the "same or similar functions" clause predates 2017. |
| Program Offering Guide, current | [mathworks.com/help/pdf_doc/offering/offering.pdf](https://www.mathworks.com/help/pdf_doc/offering/offering.pdf) | Unknown | Redirects to a MathWorks sign-in | **Not obtained.** The R2019b version (09/2019) above was used. |
| **MathWorks Online Services Agreement** (governs MATLAB Online) | [mathworks.com/company/trust-center/online-services-agreement.html](https://www.mathworks.com/company/trust-center/online-services-agreement.html) | "Updated September 23, 2026" | Read in full through a web-fetch tool. Direct downloads from mathworks.com are refused from this environment (Akamai "Access Denied"). | Current. The tool limits each quote to 125 characters, so long clauses appear below as fragments joined by […]. |
| MATLAB Online Versions page (basic vs full) | [mathworks.com/products/matlab-online/matlab-online-versions.html](https://www.mathworks.com/products/matlab-online/matlab-online-versions.html) | No date shown | Web-fetch tool | Current marketing page, not a contract. |
| `matlab-actions/setup-matlab` README and licence | [README.md](https://github.com/matlab-actions/setup-matlab/blob/5566ed98a05af94150b7edbe0450faac70b3250a/README.md), [license.txt](https://github.com/matlab-actions/setup-matlab/blob/5566ed98a05af94150b7edbe0450faac70b3250a/license.txt) | Commit `5566ed9` (HEAD on 26 Nov 2026) | Downloaded raw | Current. |
| `run-command`, `run-tests`, `run-build` READMEs; `matlab-actions` org profile; `setup-matlab` v1 and v2 READMEs | raw.githubusercontent.com, e.g. [run-command README](https://github.com/matlab-actions/run-command/blob/4f4147427a14343356827f9aced1b23559bfb310/README.md) | HEAD on 26 Nov 2026 | Downloaded raw | Current. They add only the same "separate terms" note. |
| MATLAB batch licensing (`matlab-batch`) | [MATLAB-BATCH.md](https://github.com/mathworks-ref-arch/matlab-dockerfile/blob/cf867453e63ee023f0ec4c30aeca25a88b6de4dd/alternates/non-interactive/MATLAB-BATCH.md) | Changelog up to v2026.08.0 | Downloaded raw | Current. |
| MathWorks CI page | [mathworks.com/solutions/continuous-integration.html](https://www.mathworks.com/solutions/continuous-integration.html) | No date shown | Web-fetch tool | Current marketing page. |
| Batch Licensing Pilot page | [mathworks.com/support/batch-tokens.html](https://www.mathworks.com/support/batch-tokens.html) | No date shown | Web-fetch tool | Current. The form itself was not visible. |
| GNU GPL v3 (GNU Octave's licence) | Local copy `/usr/share/common-licenses/GPL-3` (gnu.org was unreachable from here) | GPL v3 | Read | Standard text. |

---

## 3. Clauses, document by document

Quotations are word for word. Section numbers are the documents' own.
Comments in *italics* are mine.

### 3.1 MathWorks Software License Agreement (03/2017 text)

Source: [R2019b copy](http://mib.helsinki.fi/Matlab_license.txt), checked
against the [R2020a copy](https://zim.uni-wuppertal.de/fileadmin/zim/dienste/pdf/MathWorks_SLA_R2020a.pdf).
*This is not the current agreement. It is the newest full text I could
obtain without running MATLAB.*

**Lead-in to the restrictions** (Article 3): "Licensee shall not, and shall
not cause or permit any other individual or entity to, directly or
indirectly:"

#### (a) Developing, testing or benchmarking a competing, substitute or compatible product

> **3.2.** use a Program for any act which infringes copyright of a Program, for instance, by developing, producing, or testing a computer program containing a feature or functionality that is substantially similar in its expression to the expression contained in a Program;

> **3.4.** adapt, translate, copy, convert, use, test, benchmark, or determine the expression or design of, a Program or any materials derived from or generated by a Program, in order to make or distribute an application or software, a principal purpose of which, as reasonably determined by MathWorks, is to perform the same or similar functions as a Program or other MathWorks product or service or which replaces any component of a Program or other MathWorks product or service. Licensee shall not use a Program to compete with the products or businesses of MathWorks, including by distributing libraries or any form of an entire Program or Program Component;

> **3.23.** create an Application (a) providing capability similar to the capabilities of any of the Programs; (b) providing functionality or behavior similar to that of the MATLAB command line or scripting and programming capabilities; (c) allowing operation of the code generation capabilities of Programs; (d) providing access to an entire Program or a substantial portion of a Program; or (e) enabling evaluation of end-user code or expressions;

Definitions these depend on:

> **23.17.** "Program" means the software licensed hereunder, including Program Components, Documentation, enhancements, and error corrections. Each product licensed hereunder is a separate Program.

> **23.6.** "Documentation" means the user guides, if any, accompanying delivery of a Program or made available at mathworks.com/help, as may be updated from time to time, including the bug reports made available at mathworks.com/support/bugreports and the Program Offering Guide. […]

> **23.2.** "Application" means binary executable software that runs independently of the Programs, consisting of one or more software files that Licensee has created by either (a) using a Program to compile or translate Licensee created Source Code or (b) incorporating or linking any part of any compiled Source Code, library file, or other Program Component provided with the Program. […]

*The words "compatible" and "interoperable" do not appear. In this text
"benchmark" appears only in 3.4.*

*Earlier form (R2014a-era copy, its §5.3): "adapt, translate, copy, or
convert all or any part of a Program in order to create software, a
principal purpose of which is to perform the same or similar functions to a
Program or to replace a Program or any component of a Program;". The 2017
text added "use, test, benchmark" and "materials derived from or generated
by a Program".*

#### (b) Publishing benchmarks or comparisons

*No clause restricts publishing benchmark results or comparisons as such.
The nearest clauses are:*

> **3.13.** republish the Documentation, except as expressly permitted in Article 2 of this Agreement;

> **9. PUBLICITY.** Licensee may not use MathWorks' name, trade names, logos, or other trademarks of MathWorks or any of its Affiliates or Licensors in any advertising, promotional literature or any other material, whether in written, electronic, or other form, distributed to any Third Party, except in the form provided by MathWorks, and then solely for purposes of identifying MathWorks' Programs.

#### (c) Reverse engineering, decompiling, behavioural analysis

> **3.3.** disassemble, decompile, or reverse engineer a Program, or attempt to gain access to its method of operation or source code;

> **3.15.** disable, circumvent, or workaround any technical limitations in the Programs;

> **3.16.** access, enable access to, modify, translate, use or otherwise make available temporary intermediate files produced by a Program;

> **12. EUROPEAN UNION SOFTWARE DIRECTIVE.** For any Licensed User properly licensed to use the Programs within the European Union, any contractual provisions of this Agreement contrary to laws implemented pursuant to Article 6 of Directive 2009/24/EC of the European Parliament and of the Council of 23 April 2009 on the legal protection of computer programs or to the exceptions provided for in Article 5(2) and (3) of such Directive shall be null and void solely to the extent decompiling, disassembling, or otherwise reverse-engineering of the Programs is necessary to enable the Licensee to achieve the interoperability of an independently created program with the Programs or any other permitted objectives specified by such laws implemented under such Directive (collectively, the "Permitted Objectives"), consistent with the Directive, provided that any such information gained is used solely for such Permitted Objectives and solely in the European Union.

*"Behavioural analysis" is not named. Black-box testing is reached through
the words "use, test, benchmark, or determine the expression or design of"
in 3.4.*

#### (d) Permitted purpose and eligible users

> **2.1.** install and use the Programs solely on Computers controlled by Licensee, in accordance with the License Offering acquired, and solely for Internal Operations;

> **4.** […] Licensee agrees to use any Program provided to Licensee as a trial evaluation only for the period of the trial evaluation which may be controlled by a license key code, only to evaluate it individually for potential purchase of a license to the Program as an end-user, to conduct no business with it, […] Licensee agrees to use any Program awarded to Licensee on a complimentary basis for student engineering competitions and similar competitions, course support, research, fellowship participation, and teaching assistance only for the specific purpose and period of the award, to conduct no business with it, and to comply with all other obligations and restrictions in the award letter and in this Agreement.

From the R2019b **Program Offering Guide** (09/2019), Part One:

> **2.3 Student (iii)** A Student License may not be used for activities that are commercial, for profit, professional (except for primary and secondary school teachers using the License for the purpose of teaching at such schools), government, or revenue-generating.

> **2.4 Home (i)** A Home License may be licensed by an individual and may be used only for personal use. A Home License may not be used for activities that are commercial, for profit, professional, academic, government, or revenue-generating.

> **Glossary, "Academic Internal Operations"** […] for the purpose of (i) in the case of employees and other academic staff, performing software administration, teaching, and noncommercial, academic research in their ordinary course; and (ii) in the case of enrolled students, meeting classroom requirements of courses and study offered by the Licensee. […]

*No category for open-source projects appears. MATLAB Online is mentioned
only as a Software Maintenance Service benefit: "Access and storage for
online services including MATLAB Mobile, MATLAB Drive, and MATLAB Online".*

#### (e) Automated, scripted or CI use

> **3.14.** create a server or automated system for the Transformation Programs without obtaining a license from MathWorks that expressly allows such use (see the Program Offering Guide);

> **3.19.** mechanize or automate the process of checking out or in license keys for one or more Programs, including by running a second session of a Program, running a Program to serve multiple users, or rebooting the License Manager, for a principal purpose of minimizing the License check out time of any Programs, or otherwise circumventing the intended License Manager operation;

Program Offering Guide (09/2019), Part Three:

> "Automation Server" means an automated, batched, cron, scheduled, or unattended process for software development, simulation, code generation, validation, verification, or deployment or any other type of Program execution that processes Program Input Artifacts. Automation Server includes server farm, cluster, grid, build system, continuous integration system, regression server, data center, and similar types of systems. […]

> **2(iv)** Only Licensed Users of a Program may create or modify Program Input Artifacts that are processed by that Program running on an Automation Server.

> **2(viii)** Programs licensed under Individual or Network Named User License Option: Programs may be run on an Automation Server or from an External Application if the Program license is dedicated to this use and is not used for other purposes, and the Program is not used on more than one (1) Computer at any time.

#### (f) Breach: termination and liability

> **8.3.** […] MathWorks may terminate this Agreement and all Licenses granted hereunder by written notice to Licensee if Licensee breaches any material term of this Agreement, including failure to pay any License fees due, and Licensee has not cured such breach within sixty (60) days of written notification. […] MathWorks may immediately terminate upon notice this Agreement and all Licenses granted hereunder should Licensee breach the terms and conditions of Articles 2, 3, 6, and/or 10. […]

> **8.5. Survival.** The following provisions survive any termination of this Agreement: Articles 1, 3, 6, 8-11, and 13-23.

> **17.** […] Licensee will defend, indemnify, and hold harmless MathWorks and its Licensors, officers, directors, employees, agents and resellers from and against any damages, liabilities, costs and expenses (including reasonable fees of MathWorks' attorneys) arising out of any Third Party claim or demand based on or arising from, out of or in connection with (i) the creation, use, or distribution of any Applications by Licensee or any of its Affiliates or (ii) the use of the Programs by Licensee or any of its Affiliates in breach of this Agreement.

> **18.** This Agreement shall be interpreted, enforced and construed and governed in all respects by the laws of the State of Delaware, United States of America, without regard to its conflicts of law provisions. […]

> **19. COMPLIANCE WITH LICENSES.** Licensee agrees to notify MathWorks promptly upon discovery of any failure to comply with this Agreement. Licensee agrees that, no more than once every twelve (12) months or within a reasonable time after a transfer, in addition to any other remedies at law or in equity, MathWorks or its authorized representative shall, upon reasonable prior notice, have the right to inspect Licensee's and Licensee's Affiliates' records, systems, and facilities, […]

*A breach of Article 3, which contains 3.4, allows immediate termination
(8.3). Article 3 survives termination (8.5). Article 15 caps MathWorks'
liability to the licensee; I found no clause in this text capping the
licensee's liability to MathWorks.*

### 3.2 MathWorks Online Services Agreement (updated 23 September 2026)

Source: [online-services-agreement.html](https://www.mathworks.com/company/trust-center/online-services-agreement.html).
*Quotes are capped at 125 characters by the fetch tool, so long clauses
appear as fragments joined by […]. Check them against the live page.*

Scope:

- Introduction: "IF YOU DO NOT USE THE SERVICES, THIS AGREEMENT DOES NOT APPLY TO YOU."
- Definition: "Services" means "the specific MathWorks services listed in Section 5 below".
- Section 5 lists 5.1 MATLAB Answers, 5.2 File Exchange, 5.3 ThingSpeak,
  **5.4 MATLAB Online and Simulink Online**, 5.5 Online Training Courses,
  5.6 MATLAB Drive, 5.7 Cody, 5.8 MATLAB Mobile, 5.9 Discussions,
  5.10 Cloud Center, 5.11 AI Chat Playground, 5.12 MATLAB and Simulink
  Integration for GitHub, and 5.13 MATLAB Course Designer. *None of them is
  the public documentation at mathworks.com/help. 5.12 is about MathWorks
  storing repository content "for the purpose of providing users with file
  viewing and differences"; it does not cover GitHub Actions or CI
  licensing.*
- **5.4** (a single sentence): "Your use of MATLAB Online and Simulink Online is subject to this Agreement" […] "if applicable, the MathWorks software product license that allows you to access this Service."

**Lead-in to the restrictions** (§3.2): "You agree not to, and shall not
cause or permit any third party to, directly or indirectly:"

#### (a) Developing, testing or benchmarking a competing or similar product

- **§3.2(3):** "adapt, translate, copy, convert, use, test, benchmark, or determine the expression or design of, a Service" […] "in order to make or distribute a service or other application, a principal purpose of which, as reasonably determined by MathWorks, is to perform the same or similar functions as any of the Services" […] "which replaces any component of any of the Services" […] "or any other technology, materials, products, or services licensed or provided by MathWorks"
- **§3.2(2)** (copyright): "by developing, producing, or testing a service or content containing a feature or functionality" […] "substantially similar in its expression" […] (to a Service).
- **§3.2(8)** (copyright): "developing, producing, or testing software containing a feature or functionality that is substantially similar" […] (to MathWorks software).
- **§2.6** is a list that begins "Your Content must not (i) be libelous, defamatory, abusive, threatening, harassing, offensive" […]; its item (iii) is "directly compete with products or services offered by MathWorks". *This governs content you upload or post, not how you use the Services.*

*Unlike SLA 3.4, §3.2(3) has no separate sentence about competing with
MathWorks' businesses. "Compatible" and "interoperable" do not appear.*

#### (b) Publishing benchmarks or comparisons

*No clause. "Benchmark" appears only in §3.2(3). "Publish" is never used
about results, performance or comparisons.* The nearest clauses:

- **§3.2(1):** "republish, or otherwise exploit all or any part of the Services unless explicitly permitted by MathWorks"
- **§3.2(5):** "sell, license, sublicense, provide access, publish, display, distribute" […] (the Services, to third parties, without consent).

#### (c) Reverse engineering and probing

- **§3.2(6):** "disassemble or reverse engineer any of the Services" […] "attempt to gain access to any method of operation or source code contained therein"
- **§3.2(13):** "circumvent, disable, or otherwise interfere with any technical and/or other limitations"
- **§3.2(14):** "nor attempt to gain unauthorized access to any feature of the Services"

#### (d) Permitted purpose and eligible users, including the free basic tier

- **§3.1:** "personal, non-exclusive, non-transferable, limited, revocable, and temporary authorization to access and use the Services" […] "Any other use is expressly prohibited by this Agreement."
- **§1:** "You will need a MathWorks Account to use the Services"
- **§6.1:** users must be "over the age of 13, as the Services are not intended for children under 13" (with a parental-consent route).
- *There are no academic, commercial or open-source eligibility rules and
  no separate terms for the free tier.* The tier is described only on the
  [Versions page](https://www.mathworks.com/products/matlab-online/matlab-online-versions.html),
  which is not a contract: "A basic version of MATLAB Online provides 20 hours per month of free use and access to 10 commonly used products." and "all users with a MathWorks Account will have access to the basic version of MATLAB Online by default." Its table lists 15 minutes of continuous compute for the basic tier.

#### (e) Automated, scripted or CI use

*No clause on automated, scripted or bot access, and none on access "by
means other than" MathWorks' interface.* The nearest clauses:

- **§3.2(14):** "use the Services in any manner that could damage, disable, overburden, disrupt, or impair any MathWorks systems"
- **§3.2(4):** "or otherwise provide or allow access (directly or indirectly) to any of the Services to any third party"

#### (f) Breach: suspension, termination and liability

- **§3.3:** "MathWorks reserves the right to temporarily suspend or permanently terminate your access to any or all Services" and "If you breach this Agreement, your authorization to use the Services automatically terminates."
- **§1:** "MathWorks may delete your MathWorks Account after a period of inactivity or if you breach this Agreement" […] "may result in loss of access to the Services and Your Content"
- **§6.4:** MathWorks will not "be liable to you for any direct, indirect, incidental, special, punitive, or consequential damages whatsoever" […] and its "aggregate liability arising out of or related to this Agreement exceed US one hundred dollars ($100.00 USD)" (that is, it will not exceed that).
- **§6.5:** "You agree to indemnify and hold harmless MathWorks" […] *The claims it covers include breach of the Agreement; this is my reading of text I could not quote in full.* The indemnity survives termination.
- **§6.6:** "governed, in all respects by the laws of the State of Delaware"
- **Introduction:** "MathWorks may, in its sole discretion, change this Agreement at any time by making the revised Agreement available to you."

### 3.3 The free MATLAB licence in GitHub Actions

Source: [setup-matlab README, "Licensing" and "Notes"](https://github.com/matlab-actions/setup-matlab/blob/5566ed98a05af94150b7edbe0450faac70b3250a/README.md#licensing)
(commit `5566ed9`).

> Product licensing for your workflow depends on your project visibility:
>
> - Public project — The Run MATLAB Build, Run MATLAB Tests, and Run MATLAB Command actions automatically license all products for you, except for transformation products, such as MATLAB Coder and MATLAB Compiler.
> - Private project — The actions do not automatically license any products for you.

> - Public project and MATLAB batch licensing do not support external language interfaces, including MATLAB Engine APIs for Python®, Java, .NET, COM, C, C++, and Fortran. […]
> - When you use the **Setup MATLAB** action, you execute third-party code that is licensed under separate terms.

The v1 README said: "Currently, it is available only for public projects."
The run-command, run-tests and run-build READMEs and the org profile repeat
the "separate terms" sentence. None of them links to those terms.

The action's own [license.txt](https://github.com/matlab-actions/setup-matlab/blob/5566ed98a05af94150b7edbe0450faac70b3250a/license.txt)
is a BSD-3-Clause licence ("Copyright 2022 The MathWorks, Inc.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met: […]"). *It covers the action's JavaScript, not MATLAB.*

[MathWorks CI page](https://www.mathworks.com/solutions/continuous-integration.html):

- "You can run MATLAB and Simulink on your own CI server or use a cloud-hosted CI service if your project is shared publicly."
- "MATLAB, Simulink, and most add-on products are available at no additional cost for batch-use-only in Azure DevOps, CircleCI, or GitHub® Actions."

[MATLAB-BATCH.md](https://github.com/mathworks-ref-arch/matlab-dockerfile/blob/cf867453e63ee023f0ec4c30aeca25a88b6de4dd/alternates/non-interactive/MATLAB-BATCH.md):
"The MATLAB batch licensing project is still in the pilot phase." Its
option table lists `-displayLicenseAgreement`: "Flag for showing the
software license."

[Batch Licensing Pilot page](https://www.mathworks.com/support/batch-tokens.html):
"The batch token pilot program is not currently accepting new requests."

**Coverage of the six topics:** none of these documents says anything about
competing products, benchmarks, reverse engineering, eligibility beyond
"public project" / "shared publicly" (neither term is defined), or breach.
*The terms that govern MATLAB in this setting are the "separate terms"
shipped with the software. They can be read only by installing MATLAB or
running `matlab-batch -displayLicenseAgreement`, which this review did not
do.*

### 3.4 GNU GPL v3 (GNU Octave), for comparison

> **2. Basic Permissions.** […] This License explicitly affirms your unlimited permission to run the unmodified Program. The output from running a covered work is covered by this License only if the output, given its content, constitutes a covered work. […] You may make, run and propagate covered works that you do not convey, without conditions so long as your license otherwise remains in force.

---

## 4. The four questions

### Q1. Would running the validation kit on MATLAB Online's free tier be permitted?

**What the text says.** MATLAB Online is a Service under the Online
Services Agreement (§5.4). Use of it is "subject to this Agreement" and,
"if applicable", a software licence. §3.2(3) forbids using, testing or
benchmarking a Service "in order to make or distribute a service or other
application, a principal purpose of which, as reasonably determined by
MathWorks, is to perform the same or similar functions as any of the
Services", or one that "replaces any component of" a Service or other
MathWorks product. §3.1 permits only a "personal, … revocable"
authorisation and says "Any other use is expressly prohibited". Nothing in
the Agreement or on the Versions page carves out the free tier.

**My reading.** Not permitted, on a plain reading.

- The kit's stated purpose ([validation/README.md](../validation/README.md)
  lines 3–6) is to run mylab's checks in MATLAB "so that every disagreement
  can be found and fixed". That is using and testing the Service in order
  to make mylab.
- mylab's principal purpose is to run MATLAB-language code. Its browser app
  (editor, command window, workspace, plots) does much the same job as
  MATLAB Online itself. "As reasonably determined by MathWorks" leaves the
  judgement to MathWorks.
- If MATLAB Online also brings in the SLA (5.4 "if applicable"), SLA 3.4
  applies as well, with its explicit "shall not use a Program to compete"
  sentence.
- A weaker, secondary point: the kit stops at 12 minutes and resumes, to
  stay under the 15-minute compute limit. That respects the limit rather
  than defeating it. But SLA 3.15 ("workaround any technical limitations")
  and OSA §3.2(13) ("circumvent … any technical and/or other limitations")
  are worded broadly, so it is worth not designing around the limit at all.
- Consequence on breach: authorisation "automatically terminates" (§3.3).
  The MathWorks account can be deleted (§1). The user indemnifies MathWorks
  (§6.5). That falls on whoever runs the kit, which today would be a mylab
  maintainer or user following our README.

### Q2. Would running it in GitHub Actions with the free licence be permitted?

**What the text says.** The setup-matlab README says products are licensed
automatically for a "Public project", and that MATLAB is "licensed under
separate terms", which it does not link. MathWorks' CI page offers MATLAB
"at no additional cost for batch-use-only" for projects "shared publicly".
No document I could reach states the terms of this licence, defines
"public project", or adds any exception. Every SLA text I could read has
§3.4, and the 2019b Program Offering Guide explicitly treats a "continuous
integration system" as an Automation Server.

**My reading.** Not permitted, and the uncertainty runs against us, not for
us.

- The licence is a free grant tied to public projects. The most likely
  governing text is the SLA in the installed release. Nothing suggests the
  free CI grant relaxes §3.4 for a project whose principal purpose is to do
  what MATLAB does.
- The mylab repository is public, so the workflow, its logs and any
  uploaded results would be public. That would be a visible record of
  exactly the use §3.4 and §3.2(3) describe.
- The only way to read the actual terms is to install MATLAB or run
  `matlab-batch -displayLicenseAgreement`. That was out of bounds for this
  review and would itself mean accepting the terms.
- The batch-token route (for private repositories) is closed: "not
  currently accepting new requests".

### Q3. Does "competing product" plausibly cover a free, MIT-licensed, MATLAB-compatible engine positioned as an alternative?

**What the text says.**

- SLA 3.4 has two limbs. One: "a principal purpose of which … is to
  perform the same or similar functions as a Program or other MathWorks
  product or service or which replaces any component of" one. Two:
  "Licensee shall not use a Program to compete with the products or
  businesses of MathWorks".
- OSA §3.2(3) has the first limb, extended to "any other technology,
  materials, products, or services licensed or provided by MathWorks".
- Neither clause depends on price, licence or commercial intent, and
  neither has an exception for free or open-source software.
- SLA 3.23(b) separately treats "functionality or behavior similar to that
  of the MATLAB command line or scripting and programming capabilities" as
  something a licensee's Applications may not provide.

**My reading.** Yes, plausibly, and strongly so.

- The test is functional ("same or similar functions"), not commercial.
  An engine whose purpose is to run MATLAB code is about as close a fit as
  exists.
- The project's own framing removes any doubt about intent. README line 4
  says it is for people "who can't afford a MATLAB license".
  [docs/launch/LINKEDIN.md](launch/LINKEDIN.md) line 9 says "So I built a
  free alternative".
- Being free arguably makes the "compete with the … businesses of
  MathWorks" limb fit better, not worse, because the stated aim is to serve
  people who would otherwise need a licence.
- SLA 3.23 itself applies only to "Applications" as the SLA defines them
  (binaries built with a MathWorks Program), which mylab is not. Its
  wording shows how MathWorks regards a MATLAB-like scripting engine.

Legal background (from general knowledge, **not fetched or checked for
this review**; ask counsel):

- The cases in [CLEANROOM.md](../CLEANROOM.md) (*Google v. Oracle*;
  *SAS Institute v. World Programming*) are about copyright. Contract
  terms like 3.4 are a separate question.
- In the US, courts have enforced licence terms that forbid reverse
  engineering (for example *Bowers v. Baystate*, Fed. Cir. 2003).
- In the EU, Directive 2009/24/EC lets a lawful user "observe, study or
  test" a program to find the ideas and principles behind it, and makes
  contrary contract terms void. *SAS v. WPL* applied this to a licensee
  studying a learning edition of SAS. SLA Article 12 acknowledges the
  Directive in narrower terms.
- That could be a defence for an EU-based licensee who runs black-box
  tests. It is a defence to argue later, not a permission granted in
  advance. It does not obviously cover a free online service account, and
  it does not help outside the EU.

### Q4. Do the README, the report page or the docs publish comparisons with MATLAB in a way any clause restricts?

**What the text says.** No clause I read restricts publishing benchmarks or
comparisons as such. The restrictions are on the act of using, testing or
benchmarking MATLAB to make a similar product (SLA 3.4, OSA §3.2(3)). Other
restrictions cover republishing Documentation (SLA 3.13) or the Services
(OSA §3.2(1), (5)). SLA Article 9 limits a licensee's use of MathWorks'
names in materials to "identifying MathWorks' Programs".

What the repository said when reviewed (commit `4d2f4dd`). The validation
instructions, the README invitation, the COMPATIBILITY pointer and the
CLEANROOM line have since been replaced:

| Where | What it says | Based on running MATLAB? | Clause engaged |
|---|---|---|---|
| [README.md](../README.md) line 175 | Interpreted loops are "fine for coursework, but slower than MATLAB." | No. Line 180 says "It has not been timed against MATLAB." | None. *It is an unmeasured claim, so it is an accuracy question, not a licence one.* |
| README.md lines 177–180 | Speed is compared with GNU Octave 10.3 | No | None (GPLv3 §2). |
| README.md lines 230–233 | "mylab has not been run side by side with MATLAB itself. If you have MATLAB, validation/ runs the same checks there…" | Invites it | Points readers to the use in Q1–Q2. |
| [validation/README.md](../validation/README.md) lines 29–31, 44; [validation/matlab/README.txt](../validation/matlab/README.txt) line 14; `run_mylab_validation.m` line 14; `build_kit.py` line 56 | Step-by-step instructions to run the kit on MATLAB Online with a free account, sized to the basic tier's 15-minute limit | Yes, by whoever follows it | **OSA §3.2(3); SLA 3.4 if it applies.** This is the main issue. |
| "The report page": [validation/compare_with_matlab.py](../validation/compare_with_matlab.py) lines 97, 148 | Writes `validation/report.md` titled "mylab compared with MATLAB <release>", listing every difference. The file is not in `.gitignore`. | Yes; it is built from MATLAB's output | Publishing it is not itself restricted. But it would be public evidence of the use in Q1–Q2, and SLA 3.4 also covers "materials … generated by a Program". **No report exists in the repository today.** |
| [docs/COMPATIBILITY.md](COMPATIBILITY.md) lines 25–26, 202–204 | Says nothing has been run or timed against MATLAB | No | None. |
| [datasets/matlab-tasks/README.md](../datasets/matlab-tasks/README.md) lines 117–127 | Invites people to "publish results" and "compare engines", and asks them to report tasks that behave differently on MATLAB | Invites it | Engaged only if someone runs MATLAB to do it. |
| [.github/ISSUE_TEMPLATE/report-a-difference.yml](../.github/ISSUE_TEMPLATE/report-a-difference.yml) lines 35–40 | Asks reporters to paste "What MATLAB printed", with release | Yes, by the reporter | Binds the reporter's own licence (see section 5). |
| [CLEANROOM.md](../CLEANROOM.md) line 20 | Says contributors may learn behaviour by "observing the behaviour of MATLAB or GNU Octave as a black box" | Invites it | **Conflicts with SLA 3.4 and OSA §3.2(3) for MATLAB.** It is fine for Octave. |
| [docs/launch/LINKEDIN.md](launch/LINKEDIN.md) line 9 | "a free alternative" | No | Not restricted. Relevant to Q3 because it shows positioning. |
| Web app "Publish report" (`web/js/report.js`) | Publishes a user's own code and output | No | None. It contains no comparison with MATLAB. |

**My reading.** The published comparisons are fine as they stand. They
compare with Octave and say plainly that MATLAB was never run, and no
clause restricts publishing a comparison. The problem is the instructions:
the validation material, the README's invitation and the CLEANROOM line
tell people to do what the terms forbid. A future `validation/report.md`
would publicly record it.

SLA Article 9 binds only licensees. The README's trademark notice already
uses "MATLAB" only to describe compatibility, so I see no issue there.

---

## 5. Recommendation

| Route | Rating | What to do |
|---|---|---|
| MATLAB Online free tier | **Unsafe** | Don't use it for mylab validation, and don't tell others to. |
| GitHub Actions free licence | **Unsafe** | Don't add a `setup-matlab` workflow. Don't install MATLAB in CI to read its terms either, since installing it means accepting them. |
| Other MathWorks licences, trials, batch tokens | **Unsafe** | Same SLA 3.4. Home and Student licences also exclude academic or professional use, and trials are limited to evaluating a purchase. |
| Documented behaviour + GNU Octave | **Safe** for non-licensees; **uncertain** for licensees | The maintainer holds no MathWorks licence. Keep this as the primary method (it is what produced 700/705 and 469/469). People who hold a MathWorks licence should know that SLA 23.17 makes the Documentation part of the "Program". |
| User-submitted difference reports | **Uncertain** | Keep accepting them, but don't solicit systematic runs. Prefer reports that cite the documented behaviour. |
| Ask MathWorks | **Safe** to ask | Do it if MATLAB validation matters enough. |

Concrete steps. Steps 1–3 are done; steps 4–6 are still open.

1. **Keep MATLAB validation on hold.** *(Done.)* No MATLAB Online runs and no
   `setup-matlab` workflow. Nothing in `.github/workflows/` uses MATLAB
   today. I checked.
2. **Pause the invitation.** *(Done: the kit's README files and comments,
   README, COMPATIBILITY, CLEANROOM and CHANGELOG now say the kit is on hold
   and was never run. CLEANROOM forbids running MATLAB to test mylab.)*
   Replace the MATLAB Online instructions in
   `validation/README.md` and `validation/matlab/README.txt` with an
   "on hold pending licence review" note, or remove the kit. Reword README
   lines 230–233 and the "If you have MATLAB" sentence in
   `docs/COMPATIBILITY.md` lines 9–11 to match. Remove "observing the
   behaviour of MATLAB … as a black box" from CLEANROOM.md line 20 and keep
   it for Octave only.
3. **Don't publish a MATLAB-generated report.** Add `validation/report.*`
   to `.gitignore` while the kit exists. *(Done.)*
4. **Steer difference reports.** In `report-a-difference.yml`, make the
   documented-behaviour path the first option: what you expected, plus a
   link to the MathWorks reference page. Make "what MATLAB printed"
   optional, with a line asking reporters to make sure their own licence
   allows sharing it. A one-off report from someone who hit a difference in
   their own work is far lower risk than a test campaign we ask them to run.
   Note, from general knowledge and not verified: systematically
   encouraging licensees to breach their licence can create
   inducement-to-breach risk for the project. *(Done: the form now asks for
   the expected result and a documentation link first; MATLAB output is
   optional, with a licence reminder.)*
5. **If MATLAB validation is worth having, ask MathWorks in writing.** SLA
   22 requires "a written agreement signed by MathWorks and Licensee" to
   change the SLA. Both 3.4 and §3.2(3) leave the judgement "as reasonably
   determined by MathWorks". Describe the kit exactly: what it runs, that it
   reads no MathWorks code, and what would be published. Ask for permission
   for a named route (desktop licence or CI). Assume the answer may be no,
   and go ahead only on a written yes.
6. **Get a lawyer's view** before relying on any of the readings above,
   especially the EU-law point in Q3 and the status of documentation use
   for contributors who hold MathWorks licences.

---

## 6. What could not be fetched or verified

- **The current SLA.** `mathworks.com/license` returns 404. MathWorks
  states that the agreement is distributed only as `license_agreement.txt`
  in an installation (or, before R2025a, the Help menu). Reading it means
  running MATLAB or MATLAB Online, which this review did not do. The SLA
  quotes above come from **third-party mirror copies of the 03/2017 text**
  (R2019b and R2020a installers). I cannot say whether the current (2026)
  text differs.
- **The current Program Offering Guide.** The PDF needs a MathWorks
  sign-in. The 09/2019 version (R2019b) was used.
- **The terms of the free GitHub Actions licence.** No public document
  states them. They can be read only from an installed MATLAB or
  `matlab-batch -displayLicenseAgreement`, which was not done.
- **Which licence governs the MATLAB software inside a free MATLAB Online
  session.** OSA §5.4 says "if applicable" and names none.
- **The full Online Services Agreement wording.** It was read through a
  web-fetch tool that limits each quote to 125 characters. Direct downloads
  from mathworks.com are refused from this environment (Akamai "Access
  Denied"), and web.archive.org was unreachable. The §6.5 indemnity scope
  is paraphrased. The fetch tool's own summary mis-numbered some sections;
  the section numbers here come from the agent's reading of the page text.
  Check the quotes against the live page.
- **Batch-token eligibility.** The form on the pilot page was not visible.
  The page says the pilot is not accepting new requests.
- **MathWorks Store "Terms and Conditions of Sale".** The page body did not
  load (it seems to render with JavaScript).
- **Terms of use for the public documentation site** (mathworks.com/help).
  None was found. The Online Services Agreement does not list it as a
  Service.
- **Legal background** (*Bowers v. Baystate*, *SAS v. WPL*, Directive
  2009/24/EC Articles 5(3) and 8, inducement to breach) is from general
  knowledge. None of it was fetched or checked for this review.
- **github.com pages** could not be opened directly from this environment.
  The matlab-actions READMEs were read from raw.githubusercontent.com at
  the commits listed in section 2.
