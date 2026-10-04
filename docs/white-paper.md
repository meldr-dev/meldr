# Meldr: data in, harmonized data out

## A white paper for a reader who starts with zero knowledge

## How to read this paper

This paper explains one software library. The library has the name Meldr.

You do not have to know data engineering. You do not have to like the idea. A reader who doubts Meldr is welcome.

Parts 1 to 5 explain the problem and the idea. Parts 6 to 8 show how Meldr works. Part 9 gives the best arguments against Meldr, and our answers. Part 10 lists what we did not prove. Part 11 shows how to test Meldr in one day.

## 1. The short version

Companies get data from many systems. Each system describes the same thing in a different way. A person must change all of it into one shape before the company can use it.

Meldr does this work. It reads data from tables, files, and documents. It changes the data into one shape that you define. It puts bad data in a separate place and tells you the cause.

Meldr uses a large language model (LLM) in one place only. The LLM suggests how to change the data. A person approves the suggestion. After that, ordinary code does the work. The LLM does not touch your rows.

Meldr does not fetch data. It does not run a schedule. It does not store your data unless you tell it to. Your system gives data to Meldr, and Meldr gives data back.

## 2. A small story

A company sells software to builders. It has three customers. Each customer sends a list of its suppliers.

Customer A sends a file with the columns `VNDR_NM` and `REV_CENTS`. Customer B sends a file with the columns `supplier_name` and `revenue_usd`. Customer C sends a file with the columns `Company` and `Annual Revenue ($)`.

All three files hold the same facts. A supplier has a name and a yearly revenue. But the names of the columns are different.

The units are different too. Customer A counts money in cents. Customer B counts money in dollars. Customer C writes the number as text, with a dollar sign and commas.

The company wants one list. It wants one column called `legal_name` and one column called `annual_revenue`. The numbers must all use dollars.

Someone must write code for each customer. The code must rename the columns. It must remove the dollar signs. It must divide the cents by 100. Then a new customer joins, and the work starts again.

## 3. What data harmonization means

Data harmonization means that you change data from many shapes into one agreed shape.

The agreed shape has a name. In Meldr, the name is a **canonical entity**. An entity is one kind of thing in the world, for example an `Organization` or a `Person`. A canonical entity says which fields the thing has, which type each field has, and which rules each field must follow.

The steps are the same each time:

- Find which source column holds which fact.
- Change the value to the correct type and unit.
- Check the result against the rules.
- Keep the rows that fail, and record the cause.

Most teams do these steps by hand. The steps are slow and they break when a source changes.

## 4. How teams do this work today

There are three common ways. Each way has a cost.

### Way 1: write the code by hand

An engineer writes SQL or Python for each source. This way is accurate and it is easy to test. But it does not scale. Ten sources must have ten pieces of code. A source that adds a column can break its code in silence.

### Way 2: ask an LLM about every row

A team can send each row to an LLM and tell it to return clean data. This way is fast to build. But it is slow and costly for large tables. The answer can change from one day to the next. The LLM can invent a value. It also sends your private data to a third party.

### Way 3: search tools for documents

Some tools cut a document into pieces and store them for search. These tools help a person find a text. They do not give you a clean table with correct types, links between rows, and a record of where each value came from.

## 5. The idea behind Meldr

Meldr splits the work into two times.

**Design time** happens one time for each source shape. Meldr looks at the column names, the types, and some sample values. The LLM suggests a mapping. A person reviews it. The approved mapping becomes a file. The file is the **spec**.

**Data time** happens for every batch. Meldr reads the spec. It applies each step with a fixed set of approved operations. It does not call the LLM. The same input and the same spec give the same output, every time.

This split has three effects:

- The LLM cost depends on the number of new source shapes. It does not depend on the number of rows.
- A person can read the spec before a row changes.
- The LLM cannot write code that runs. It can only choose from the operations that Meldr already has.

The last point is a hard rule of the design. Meldr treats LLM output as data. It checks that data against a strict format. Meldr does not run it as code.

## 6. One row from start to finish

This is one row from a source system.

```text
VNDR_ID   = V1
VNDR_NM   = Acme Corp
REV_CENTS = 100000
```

This is the approved mapping.

```text
VNDR_ID         -> org_id
VNDR_NM         -> legal_name
REV_CENTS / 100 -> annual_revenue
```

This is the output row.

```text
org_id         = V1
legal_name     = Acme Corp
annual_revenue = 1000.00
```

Each arrow in the mapping is one step. Each step uses one approved operation. A person can read the full spec in one minute.

At a different time, a second row arrives with `REV_CENTS = "n/a"`. Meldr cannot turn that text into a number. It does not put a null in the output and hide the problem. It sends the row to **quarantine** and records the cause. The rest of the batch continues.

## 7. Documents, text, and links

Not all data is a table. A row can hold a link to a PDF contract. A row can hold a long note in a text column.

Meldr can read this content and make new rows from it. You tell Meldr which columns hold links and which columns hold text. You also tell it which entity each one maps to.

```python
result = mr.harmonize(
    rows,
    target="Organization",
    documents={"contract_url": "Project"},
    texts={"notes": "Event"},
    resolve=download,
)
```

The `resolve` function is yours. It turns a link into bytes. Meldr does not open a network connection by itself. This keeps your credentials and your network rules in your own hands.

For each value that Meldr extracts, it must quote the text that supports the value. Meldr checks that the quote is in the document. If the quote is not there, Meldr rejects the value. This rule limits invented facts.

Meldr can also return a list of items from one document, for example each clause of a contract. Each item becomes a row with the ID of its parent row. Meldr can also return relations, such as "Company X owns Company Y". You list the verbs that you permit. Meldr sends a relation with a verb that is not on the list to quarantine.

## 8. What comes back

The call returns an object. It has these parts:

- `data`: the rows that pass all checks.
- `quarantine`: the rows that fail, with the cause of each failure.
- `review`: the mapping fields that a human must decide.
- `children`: rows that Meldr made from documents and text.
- `relations`: the links between things, with the quote that supports each link.
- `report`: counts, warnings, and the number of LLM calls.

Meldr does not write to your database unless you tell it to. You decide where each part goes.

The review step works like this. Meldr gives a confidence score to each field. A field above your limit passes. A field below it becomes a review item.

A person approves, edits, or rejects the item. Meldr saves a new version of the spec. The next batch uses that version, and no LLM call is necessary.

```python
with Meldr.from_config("meldr.yaml") as mr:
    result = mr.harmonize(rows, target="Organization")

clean = result.data
bad = result.quarantine
todo = result.review
```

## 9. The best arguments against Meldr

We list the strongest objections. Some of them are correct in part.

### "I can write this code myself"

For one or two sources, you can. Write the code. Meldr is not worth its cost in that case.

The cost of hand code grows with the number of sources and with the rate of change. Meldr aims at the case where you have many sources, and more sources join each month. A software company that takes data from its customers is the clearest example.

### "I already use dbt"

dbt is very good at SQL work inside a warehouse. But dbt starts after the data is in the warehouse and already has a clean shape. Someone must still write the SQL for each source. dbt does not read a PDF.

Meldr works before dbt. Meldr makes the clean table. dbt then builds your business logic on top of it. The two tools do different jobs.

### "An LLM will make mistakes"

This is true. The design assumes it. The LLM only suggests, and a person approves.

Meldr tests each suggestion on sample rows before it runs. Meldr checks every output row. A row that fails goes to quarantine.

A wrong approved mapping is still possible. A person who approves a bad spec causes a bad result. The review step lowers this risk. It does not remove it.

### "This is one more dependency"

This is true. Meldr is a Python library. The core install is light. Heavy parts, such as vector stores and media tools, are optional extras. You can use Meldr as one function call inside your current job. You do not have to move your full pipeline.

### "My data is private"

Meldr checks a data policy before it sends anything to a model. You declare what a request contains. If the policy does not permit it, Meldr stops with an error and sends nothing. You can also point Meldr at a model that runs inside your own network. Meldr does not write the values of your data to its logs. It logs field names, counts, and IDs.

### "The tool replaces my people"

It does not. It moves their time. They spend less time on rename and cast code. They spend more time on review of mappings and on quarantine.

## 10. What we did not prove

A good white paper states its limits. These are ours.

**Mapping accuracy on your data.** We measured the mapping step without an LLM. We used a public set of table pairs from actual systems (the Valentine set). The result is an F1 score of 0.68. This is the floor that the heuristic gives when no LLM is present.

We have not run a large live test with a frontier LLM on customer data. The target in the product plan is 0.90. We do not claim it.

**Speed.** On one laptop, Meldr ran and checked 1 million rows with 30 mapped fields in about 5 seconds. Profiling the same table took 0.7 seconds. These are single measurements. We have not published a repeatable benchmark script. LLM calls at design time are much slower than this.

**Security.** Meldr has tests for the main security rules. One test checks the rule that model output does not run as code. An outside security review has not happened at this time.

**Media.** Meldr has code for audio and video. The test with made media files is not complete.

**Review screen.** Meldr gives data and functions for review. It does not give a web page. A team that wants a business user to approve mappings must build the screen, or wait for one.

**Age.** Meldr is new. It has no public record of use in production.

## 11. A one-day test

You can test Meldr yourself and decide with your own data.

1. Pick one entity, for example `Supplier`. Write its fields in a short YAML file.
2. Pick three sources of that entity from your own systems. Use files with at most a few thousand rows.
3. Run `harmonize()` on the first source with the fake provider. This shows the full flow with no network.
4. Connect an LLM of your choice with a test key. Run the same source again.
5. Open the review items. Count how many fields the LLM mapped correctly.
6. Run the second and third sources. Count how many fields you had to fix.
7. Run the first source again. Make sure the report shows zero LLM calls.
8. Open `quarantine`. Decide if the causes are clear.

Compare the time with the time you spend on hand code. Use these numbers to decide.

Meldr is a poor fit in these cases:

- You have one source and it does not change.
- Your data is small.
- You cannot permit an LLM to see sample values.

Meldr is a good fit if you have many sources, if the sources change, and if part of the data is in documents.

## Glossary

- **Canonical entity:** the shared shape of one kind of thing.
- **Spec:** the saved and approved mapping plan for one source shape.
- **Transform:** one approved operation that changes a value.
- **Quarantine:** the place for rows that fail a check.
- **Lineage:** the facts that show where an output value came from.
- **LLM:** a large language model, a program that reads and writes text.
- **Design time:** the time when Meldr studies a new source shape.
- **Data time:** the time when Meldr applies a spec to a batch.
- **Resolve function:** your function that turns a link into bytes.
