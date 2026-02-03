
  COMPREHENSIVE GAP ANALYSIS: Paper vs Anonymous Repo

  1. CRITICAL: Statistical Discrepancies
  ┌───────────────────────┬───────────────────────────┬───────────────────────────┬─────────────────────┐
  │        Metric         │       Paper Claims        │         Repo Data         │       Status        │
  ├───────────────────────┼───────────────────────────┼───────────────────────────┼─────────────────────┤
  │ Anger bias χ²         │ χ²(1) = 1795.16, V =      │ χ²(1) = 1461.78, V =      │ ⚠️ MISMATCH         │
  │                       │ 0.972                     │ 0.831                     │                     │
  ├───────────────────────┼───────────────────────────┼───────────────────────────┼─────────────────────┤
  │ Overall Fleiss' κ     │ κ = 0.31 (Fair)           │ κ = 0.02 (Slight)         │ ⚠️ MAJOR MISMATCH   │
  ├───────────────────────┼───────────────────────────┼───────────────────────────┼─────────────────────┤
  │ % Agreement by        │ 68-92%                    │ 28-49% observed           │ ⚠️ Different        │
  │ subtype               │                           │                           │ metric?             │
  └───────────────────────┴───────────────────────────┴───────────────────────────┴─────────────────────┘
  Recommendation: Verify the paper's Table 1 uses the same calculation as the repo. The "% Agree" column
  may mean "≥2/3 majority" while repo shows "observed agreement" differently. The overall κ discrepancy
  (0.31 vs 0.02) is significant and needs reconciliation.

  ---
  2. Missing Outputs in Repo
  ┌────────────────────────────────┬───────────────────────────────┬─────────────────────────┐
  │         Paper Section          │        Expected Output        │       Repo Status       │
  ├────────────────────────────────┼───────────────────────────────┼─────────────────────────┤
  │ §3.4 Compositional Tests       │ Recombined scenario results   │ ❌ Not in outputs/      │
  ├────────────────────────────────┼───────────────────────────────┼─────────────────────────┤
  │ §3.5 Interventions (Table 3)   │ Intervention prediction files │ ❌ Not in outputs/      │
  ├────────────────────────────────┼───────────────────────────────┼─────────────────────────┤
  │ RSA L0/L1 baseline predictions │ L0/L1 alignment results       │ ❌ Not in outputs/      │
  ├────────────────────────────────┼───────────────────────────────┼─────────────────────────┤
  │ Table 2: Model accuracy        │ Per-model accuracy summary    │ ✅ In cogsci_stats.json │
  └────────────────────────────────┴───────────────────────────────┴─────────────────────────┘
  Recommendation: Add cached outputs for compositional tests, RSA baselines, and intervention studies so
  reviewers can verify all claims.

  ---
  3. Repo Files That Could Be Referenced in Paper
  ┌──────────────────────┬─────────────────────────────┬──────────────────────────────────────────────┐
  │      Repo File       │        Paper Status         │                Recommendation                │
  ├──────────────────────┼─────────────────────────────┼──────────────────────────────────────────────┤
  │ DATA_DICTIONARY.md   │ Not referenced              │ Add to Data Availability section or footnote │
  ├──────────────────────┼─────────────────────────────┼──────────────────────────────────────────────┤
  │ AI_DISCLOSURE.md     │ Covered by Ethics section ✓ │ Already covered                              │
  ├──────────────────────┼─────────────────────────────┼──────────────────────────────────────────────┤
  │ fig4_correlation.pdf │ Not in paper                │ Either add as Figure 4 or remove from repo   │
  ├──────────────────────┼─────────────────────────────┼──────────────────────────────────────────────┤
  │ prompts/*.txt        │ Not detailed                │ Reference in Methods or Appendix             │
  └──────────────────────┴─────────────────────────────┴──────────────────────────────────────────────┘
  ---
  4. Gap: Prompt Templates Not in Paper

  The paper (§3.5) describes interventions but doesn't show the actual prompts. The repo has
  prompts/cei_classification.txt and src/venue/intervention_prompts.py with full templates.

  Recommendation: Add a sentence like:
  "Full prompt templates are available in the supplementary repository."

  ---
  5. Potential Acceptance-Improving Revisions (No New Data Required)
  ┌────────────────────────────────────┬────────┬────────┬──────────────────────────────────────────────┐
  │              Revision              │ Impact │ Effort │                    Notes                     │
  ├────────────────────────────────────┼────────┼────────┼──────────────────────────────────────────────┤
  │ Reconcile κ discrepancy            │ HIGH   │ 1-2    │ Critical for credibility                     │
  │                                    │        │ hrs    │                                              │
  ├────────────────────────────────────┼────────┼────────┼──────────────────────────────────────────────┤
  │ Add effect sizes to all tests      │ MEDIUM │ 30 min │ Cohen's d for power gap is there, add to all │
  ├────────────────────────────────────┼────────┼────────┼──────────────────────────────────────────────┤
  │ Reference DATA_DICTIONARY.md       │ LOW    │ 5 min  │ Adds documentation completeness              │
  ├────────────────────────────────────┼────────┼────────┼──────────────────────────────────────────────┤
  │ Add sample scenario in paper       │ MEDIUM │ 15 min │ One concrete example aids understanding      │
  ├────────────────────────────────────┼────────┼────────┼──────────────────────────────────────────────┤
  │ Clarify "% Agree" definition       │ MEDIUM │ 10 min │ Table 1 footnote: "Proportion with ≥2/3      │
  │                                    │        │        │ consensus"                                   │
  ├────────────────────────────────────┼────────┼────────┼──────────────────────────────────────────────┤
  │ Add confidence intervals to Table  │ LOW    │ 15 min │ Bootstrap CIs for κ values                   │
  │ 1                                  │        │        │                                              │
  ├────────────────────────────────────┼────────┼────────┼──────────────────────────────────────────────┤
  │ Reference intervention prompts     │ LOW    │ 5 min  │ "Prompts available in repository"            │
  ├────────────────────────────────────┼────────┼────────┼──────────────────────────────────────────────┤
  │ Fix minor: "fig4" exists but not   │ LOW    │ 5 min  │ Either cite or remove                        │
  │ used                               │        │        │                                              │
  └────────────────────────────────────┴────────┴────────┴──────────────────────────────────────────────┘
  ---
  6. Specific Paper Revisions Recommended

  A. Table 1 Footnote (Clarify Agreement Metric)
  \textbf{Note:} \% Agree = proportion of scenarios where $\geq$2/3 annotators selected the same emotion.

  B. Data Availability Section (Reference Documentation)
  Current:
  "The CEI benchmark (300 scenarios), human annotations..."

  Suggested addition:
  "...including a data dictionary, prompt templates, and AI tools disclosure."

  C. Sample Scenario (Add to §3.1 or Appendix)
  Adding one concrete example from the benchmark would significantly aid reviewer comprehension. Example:
  Scenario (Strategic Politeness): A project manager says to a consistently late team member: "I really
  appreciate your creative approach to deadlines." [Gold: sadness, Models: anger]

  D. Verify/Fix χ² Statistics
  The paper reports χ²(1) = 1795.16 but repo shows 1461.78. Either:
  - The calculation method differs (check the negative-valence-only analysis)
  - The paper needs updating to match repo

  ---
  Summary Checklist

  Critical (Must Fix):
  - Reconcile overall Fleiss' κ (paper: 0.31, repo: 0.02)
  - Verify anger bias χ² matches (paper: 1795, repo: 1462)

  High Priority:
  - Add missing outputs (compositional tests, interventions, RSA)
  - Add Table 1 footnote clarifying "% Agree" metric

  Medium Priority:
  - Reference DATA_DICTIONARY.md in Data Availability
  - Add one sample scenario to Methods
  - Reference prompt templates in repository

  Low Priority:
  - Remove fig4_correlation.pdf from repo (not in paper) OR add as Figure 4
  - Add bootstrap CIs to κ values