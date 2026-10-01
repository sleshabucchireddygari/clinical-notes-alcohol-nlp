# NL-to-SQL results

Model: llama3.2:3b (local, via Ollama)

**Execution accuracy: 12/15 (80%)**

Self-corrected after an error: 3 question(s)

| Question                                                                                              | Correct   |   Attempts | LLM SQL                                                                                                                    | Error                               |
|:------------------------------------------------------------------------------------------------------|:----------|-----------:|:---------------------------------------------------------------------------------------------------------------------------|:------------------------------------|
| How many patients are in the clinic?                                                                  | yes       |          1 | SELECT COUNT(*) AS n FROM patients                                                                                         |                                     |
| How many patients were flagged for unhealthy alcohol use?                                             | yes       |          1 | SELECT COUNT(*) AS n FROM alcohol_flags WHERE flagged = 1;                                                                 |                                     |
| How many patients are on Biktarvy?                                                                    | yes       |          1 | SELECT COUNT(*) AS n FROM patients WHERE art_regimen = 'Biktarvy';                                                         |                                     |
| Which HIV medicine regimen is the most common?                                                        | yes       |          1 | SELECT art_regimen FROM patients GROUP BY art_regimen ORDER BY COUNT(*) DESC LIMIT 1                                       |                                     |
| How many female patients were flagged for alcohol use?                                                | yes       |          1 | SELECT COUNT(*) AS n FROM patient_overview WHERE flagged = 1 AND sex = 'female';                                           |                                     |
| What is the average age of flagged patients?                                                          | yes       |          1 | SELECT AVG(age) AS avg_age FROM patient_overview WHERE flagged = 1;                                                        |                                     |
| How many patients have a detectable viral load?                                                       | yes       |          1 | SELECT COUNT(*) AS n FROM patients WHERE viral_load >= 50                                                                  |                                     |
| List the IDs of flagged patients who have a detectable viral load.                                    | no        |          2 | SELECT patient_id FROM alcohol_flags WHERE flagged = 1 AND patients.viral_load >= 50                                       | no such column: patients.viral_load |
| How many flagged patients missed 3 or more doses last month?                                          | no        |          2 | SELECT COUNT(*) AS n FROM patients WHERE missed_doses_last_month >= 3 AND flagged = 1                                      | no such column: flagged             |
| What is the average CD4 count for flagged and for non-flagged patients?                               | yes       |          1 | SELECT flagged, AVG(cd4_count) AS avg_cd4 FROM patient_overview GROUP BY flagged                                           |                                     |
| What percentage of flagged patients have a detectable viral load, compared with non-flagged patients? | no        |          2 | SELECT (SUM(CASE WHEN flagged = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*)) AS percent_flagged FROM patient_overview          |                                     |
| How many patients have a CD4 count below 200?                                                         | yes       |          1 | SELECT COUNT(*) AS n FROM patients WHERE cd4_count < 200;                                                                  |                                     |
| How many patients had their last visit in August 2026?                                                | yes       |          1 | SELECT COUNT(*) AS n FROM patients WHERE last_visit_date LIKE '2026-08-%';                                                 |                                     |
| List the flagged patients older than 50.                                                              | yes       |          1 | SELECT patient_id FROM alcohol_flags WHERE flagged = 1 AND patient_id IN (SELECT patient_id FROM patients WHERE age > 50); |                                     |
| How many patients on Dovato were flagged for alcohol use?                                             | yes       |          1 | SELECT COUNT(*) AS n FROM patient_overview WHERE art_regimen = 'Dovato' AND flagged = 1                                    |                                     |

Note: matching is strict (e.g. 0.35 vs 35% counts as wrong), so read the WRONG rows - some may be reasonable answers in a different format.
