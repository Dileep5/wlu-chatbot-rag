# WLU Chatbot Expected Results

## Faculty

### Query
Who is Shohini Ghose?

Expected:
- Returns Shohini Ghose's profile.
- Includes title.
- Includes department.
- Includes research information.
- No hallucinated information.

---

### Query
Who is Tripat Gill?

Expected:
- Returns Tripat Gill's profile.
- Includes department.
- Includes research information.

---

## Department

### Query
Who works in Marketing?

Expected:
- Returns Marketing faculty.
- Uses structured retrieval.
- Does not answer with department description.

---

### Query
List Science faculty.

Expected:
- Returns Faculty of Science members.
- Shows truncation message if needed.

---

## Research

### Query
Who researches artificial intelligence?

Expected:
- Returns AI-related faculty.
- Uses research vector search.
- No unrelated faculty.

---

### Query
Who researches machine learning?

Expected:
- Returns ML faculty.
- Does not trigger conversation mode.

---

## Programs

### Query
Who coordinates the MBA?

Expected:
- Gracefully reports coordinator unavailable if database has none.
- Does not hallucinate.

---

## Courses

### Query
Who teaches CP640?

Expected:
- Current limitation.
- Should not hallucinate instructor.