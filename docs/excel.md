# EXCEL

1. **VLOOKUP or XLOOKUP to merge data**:

   * VLOOKUP: `=VLOOKUP(A2, 'Sheet2'!A:B, 2, FALSE)`
   * XLOOKUP: `=XLOOKUP(A2, 'Sheet2'!A:A, 'Sheet2'!B:B)`
2. **Difference between absolute and relative references**:

   * **Relative**: `A1`, adjusts when copied.
   * **Absolute**: `$A$1`, does not adjust.  
     *Use absolute when referencing fixed data.*
3. **Create a pivot table**:

   1. Select your data.
   2. Insert > PivotTable.
   3. Drag fields to Rows, Columns, and Values.  
      *Analysis*: Summarize, count, average, etc.
4. **Conditional formatting**:

   * Home > Conditional Formatting > New Rule.
   * Example: Highlight cells greater than 100.
5. **IF, AND, OR functions together**: