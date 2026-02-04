# NOTES

Table of contents

- [<< bit shift operator](#<<-bit-shift-operator)
- [float literals (`1.0f`) versus double literals (`1.0`)](#float-literals-10f-versus-double-literals-10)
- [Pointers and the `*` symbol](#pointers-and-the--symbol)
- [Regular variables vs pointers (`float p` vs `float *p`)](#regular-variables-vs-pointers-float-p-vs-float-p)
- [delete vs `delete[]`](#delete-vs-delete)

#### `<<` bit shift operator <a name="<<-bit-shift-operator"></a>

This is the left bit shift operator. It shifts the binary representation of a number to the left by a specified number of positions.

How it works:

1 << 20 means "take the number 1 and shift it left by 20 bit positions."

1 in binary: 0000000000000000000001

Shift left 20 positions:

                100000000000000000000
                ^
                This "1" moved 20 places left

Each left shift by 1 doubles the number (multiplies by 2):

| Expression | Binary      | Decimal         |
| ---------- | ----------- | --------------- |
| 1 << 0     | 1           | 1               |
| 1 << 1     | 10          | 2               |
| 1 << 2     | 100         | 4               |
| 1 << 3     | 1000        | 8               |
| 1 << 10    | 10000000000 | 1,024 (1K)      |
| 1 << 20    | ...         | 1,048,576 (~1M) |

So 1 << 20 = 2²⁰ = 1,048,576

This is approximately 1 million, which is why the comment says "1M elements."

Why use this instead of just writing 1000000?

- Powers of 2 are common in computing (memory sizes, buffer sizes, etc.)
- 1 << 20 makes it clear you want exactly 2²⁰
- It's a common idiom in systems programming

---

#### float literals (`1.0f`) versus double literals (`1.0`) <a name="float-literals-10f-versus-double-literals-10"></a>

For the following:

```cpp
  for (int i = 0; i < N; i++) {
  x[i] = 1.0f;
  y[i] = 2.0f;
  }
```

The f suffix indicates that these are float literals (32-bit), not double literals (64-bit).

In C/C++:

| Literal | Type   | Size   |
| ------- | ------ | ------ |
| 1.0     | double | 64-bit |
| 1.0f    | float  | 32-bit |

Why use f?

If my arrays are declared as float:
float *x = new float[N];
float *y = new float[N];

Using 1.0f and 2.0f ensures I'm assigning a float to a float, avoiding any implicit conversion from double to float.

After the above example loop:
x = [1.0, 1.0, 1.0, 1.0, ...] (1 million 1.0s)
y = [2.0, 2.0, 2.0, 2.0, ...] (1 million 2.0s)

---

#### Pointers and the `*` symbol <a name="pointers-and-the--symbol"></a>

The `*` symbol has two different meanings depending on context:

**In a declaration, means "this is a pointer"**

```cpp
float *x = new float[N];
```

Here, `*` is part of the type declaration, saying "x is a pointer to float."

**In an expression, means "dereference" (get value at address)**

```cpp
*x    // "get the value that x points to"
```

So when I write `float *x`, I'm declaring a pointer. But when I write `*x` in an expression, I'm accessing the value at that address.

Since `x` points to the first element of the array, `*x` gives me `x[0]`. To get other elements, I use pointer arithmetic:

| Array notation | Pointer arithmetic | Meaning                          |
| -------------- | ------------------ | -------------------------------- |
| `x[0]`         | `*(x + 0)` or `*x` | Value at address x               |
| `x[1]`         | `*(x + 1)`         | Value at address x + 1 position  |
| `x[5]`         | `*(x + 5)`         | Value at address x + 5 positions |
| `x[i]`         | `*(x + i)`         | Value at address x + i positions |

In fact, `x[i]` is just shorthand for `*(x + i)`; the compiler converts it internally.

---

#### Regular variables vs pointers (`float p` vs `float *p`) <a name="regular-variables-vs-pointers-float-p-vs-float-p"></a>

**Regular variable:**

```cpp
float p = 1.0f;
```

- Stores the value directly
- Lives on the stack (automatic memory)
- Automatically cleaned up when it goes out of scope

**Pointer:**

```cpp
float *p = new float;
*p = 1.0f;
```

- Stores a memory address pointing to where the value lives
- The actual value lives on the heap (dynamic memory)
- I must manually `delete` it when I'm done

When to use which:

| Regular variable           | Pointer                        |
| -------------------------- | ------------------------------ |
| Size known at compile time | Size determined at runtime     |
| Small data                 | Large data (arrays, objects)   |
| Short-lived                | Needs to outlive current scope |
| Automatic cleanup          | Manual memory management       |

---

#### `delete` vs `delete[]` <a name="delete-vs-delete"></a>

`delete[]` is specifically for arrays. For single objects, I use `delete` (without the brackets).

The rule is they must match:

| Allocation | Deallocation |
| ---------- | ------------ |
| `new`      | `delete`     |
| `new[]`    | `delete[]`   |

Examples:

```cpp
// Single object
float *p = new float;       // allocate one float
delete p;                   // free it

// Array
float *x = new float[N];    // allocate N floats
delete[] x;                 // free all of them
```

---
