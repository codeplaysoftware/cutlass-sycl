# 2D Copy Operations for Intel XE Architecture

## Overview

This document describes the **2D copy operations** introduced in CUTLASS for the **Intel XE architecture**.
These operations are designed to efficiently load and store 2D blocks of data between
**global memory** and **registers**. 


---

## Copy Operation Naming Convention

Each 2D copy operation follows this naming pattern:

```c++
XE_2D_[Packed_]<DataType>x<Rows>x<Cols>_<LD|ST>_<N|T|V>
```

Where: 

| Component            | Description                                                               |
|----------------------|---------------------------------------------------------------------------|
| `XE_2D`              | Indicates 2D copy for Intel XE                                            |
| `Packed_` (optional) | Indicates packed copy (only for `U8` and `U4` MMA operations)             |
| `<DataType>`         | Data type to load/store into registers: `U4`, `U8`, `U16`, `U32`          |
| `<Rows>x<Cols>`      | Dimensions of the 2D block in elements                                    |
| `LD` / `ST`          | `LD` = Load from global memory; `ST` = Store to global memory             |
| `N`, `T`, `V`        | Memory layout: `N` (row-major), `T` (column-major), `V` (VNNI, row-major) |

---

## 2D Copy Execution Model

All copy operations are performed at the **subgroup** level. Each **subgroup consists of 16 work-items**, 
and these **work-items** cooperate to move a 2D block of data.

### Data Distribution (Unpacked Copies)

For a load like `XE_2D_U16x32x16_LD_N`, the 32x16 matrix is split such that:

- Each **work-item gets one column** with **32 elements**

#### Example:
| Work-item | Columns Loaded |
|-----------|----------------|
| 0         | Column 0       |
| 1         | Column 1       |
| ...       | ...            |
| 15        | Column 15      |

If we increase the block size in the column dimension, for example using a `XE_2D_U16x32x32_LD_N` copy,
it will load a 32×32 block such that:

- Each **work-item gets two columns**, each with **32 elements**
- The columns are **16 apart**

#### Example:
| Work-item | Columns Loaded          |
|-----------|-------------------------|
| 0         | Column 0 and Column 16  |
| 1         | Column 1 and Column 17  |
| ...       | ...                     |
| 15        | Column 15 and Column 31 |

The same applies to the LD_V and LD_T operations.

> ⚠️ VNNI combines elements from multiple rows of a single column, packing them into 32-bit values.
It does not transform or modify the actual data, it only changes the packing format. 

### Data Distribution (Packed Copies)

When using MMA operations with **U8** or **U4** data types, such as `XE_8x16x32_S32S8S8S32_TT`, 
the data for the A matrix must be packed before being consumed by the MMA instructions. In these cases, 
a packed load operation must be used.

```c++
XE_2D_Packed_<DataType>x<row>x<col>_<LD|ST>_<N|T>
```
Example for `XE_2D_Packed_U8x32x32_LD_N`, where each **work-item gets two adjacent columns**,
each with **32 elements**

| Work-item | Columns Loaded |
|-----------|----------------|
| 0         | Column 0, 1    |
| 1         | Column 2, 3    |
| 2         | Column 4, 5    |
| ...       | ...            |
| 15        | Column 30, 31  |

---

Note: The number of adjacent columns assigned to each work-item is based on the number of columns 
in the 2D block copy operation divided by the number of work-items (16). In this case, each work-item 
loads 2 columns because the block has 32 columns. If the copy operation has 64 columns, then each 
work-item will load 4 adjacent columns.

## Supported Layout Modes

The following table summarizes the available copy variants:

| Operation  | Description                    | Matrix | Layout       | 
|------------|--------------------------------|--------|--------------|
| `LD_N`     | Load, Row-major                | A      | Row-major    |
| `LD_T`     | Load, Column-major (Transpose) | A/B    | Column-major |
| `LD_V`     | Load, VNNI layout, Row-major   | B      | Row-major    |
| `ST_N`     | Store, Row-major               | D      | Row-major    |

The `LD_V` operation is not available for the `U32` data type. Use `LD_N` instead. 

---

## Integration into GEMM Workloads

### Loading A Matrix:
- Use `LD_N` (if row-major)
- Use `LD_T` (if column-major)
- For `U8` or `U4` MMAs: use `Packed_U8` / `Packed_U4` copies

### Loading B Matrix:
- Use `LD_V` (if row-major)
- Use `LD_T` (if column-major)
- Use `LD_N` (if row-major and U32 data type)

### Storing Result Matrix:
- Use `ST_N` (row-major only)

