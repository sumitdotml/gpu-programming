# Warp drills

PMPP, fifth edition · Chapter 4, §4.4 · Figure 4.7, page 76

Practice grouping a block's threads into warps. Start with questions 1 and 2, then
check your reasoning before continuing. No answers are included.

## Setup

- Use the book's notation `T[y,x]`: row first, column second.
- Block dimensions below are **width × height**: threads across (`x`) × threads
  down (`y`).
- Assume 32 lanes per warp.
- When placing threads in linear order, `x` changes fastest, then `y`.
- Unused lanes in a partial warp are not extra executing threads.

For each question, explain your reasoning. A rough drawing is enough; focus on
thread order and warp boundaries, not neatness.

---

## 1. A 4×4 block

Use the block pictured in Figure 4.7: 4 threads across and 4 down.

- Does each row become a separate warp?
- How many warps does the whole block need?
- How many lanes are unused?

My reasoning:

If I look at Figure 4.7, I see a square-shaped logical 2D organization of threads and we are assuming that it is going to be a warp based on that linear order just below it. In this diagram, I'm checking the details. A warp could have up to 32 threads, and in here we have 4 × 4 = 16 threads. Since these are 16 threads, in this case each row has 4 threads. Those are the position 0 of y. Xs are changing, so 4 per row, I guess we could say. Since all of them fall within 32, I think each row obviously does not become a separate warp because a warp needs to have at least 32 threads. We need 32 threads for one warp, so therefore, no, each row does not become a separate warp. That's the answer to the first bullet point, assuming I'm answering for the 4x4 square organization of threads here.

for the second bullet point, for this, we need to know the block shape first. I see it now. It says a 2D block with 8x8 threads, so 64 threads. This means the whole block, if it is one block, will be one warp. Since we have two blocks, we will need two warps in total.

for the final bullet point, it says how many lanes are unused, and for this I need to try to understand what a lane is actually supposed to mean in this case. I guess a lane in this case I will understand as one thread, since we are assuming 32 lanes in one warp. We're assuming that we have a two-dimensional block with 8 by 8 threads. That means we will have 64 threads, meaning two blocks, and I would say all of the lanes will be used because all of them will be occupied by the threads and two warps in this case. Isn't that the case if we assume there are 64 threads in the two-dimensional block?

## 2. An 8×8 block

Draw the block and mark only the boundary between warps.

- Which thread is the last in warp 0?
- Which thread is the first in warp 1?

Give the thread positions using `T[y,x]`.

My drawing and reasoning:

I have drawn the 8x8 block here with 8 rows and 8 columns, starting from T(0, 0). The first row reaches T(0, 7), and the second row starts on T(1, 0) and reaches T(1, 7), and so on, up to T(7, 0), the starting point of the last row, ending with T(7, 7). The thread that is the last in warp 0, I would say, is T(3, 7), and the thread that is the first in warp 1 is going to be T(4, 0).

## 3. A 16×4 block

This block has 16 threads across and 4 down. Compare it with the 8×8 block.

- How many rows fit in each warp?
- Did the number of warps change, or just their shape in your drawing?

My reasoning:

For the 16x4 block we have 16 threads that fit inside one row across the shape (16x4). There will be 2 rows that will fit in each warp in this case. The number of warps actually did not change because 16 × 4 = 64. The block's total number of threads is 64 and one warp's capacity can be 32 lane positions. That means the number of warps didn't really change.

I don't know whether one warp can occupy more than 32 threads. By the way perhaps there is some kind of design that allows us to occupy, let's say, 64 threads or whatever. In this case from what I have studied so far, I will assume that the capacity is 32 threads in one warp. We will not really have the warp number changing here. It's just a shape in my drawing.

## 4. A 10×4 block

This block has 10 threads across and 4 down.

- Must a warp end at a row boundary?
- Where does warp 0 end, and where does warp 1 begin?

Give the thread positions using `T[y,x]`.

My drawing and reasoning:

The block has 40 threads and each row is 10 threads long. We have a total of 4 rows. A warp does not have to end at a row boundary because that does not really matter when designing warps. We placed threads into a linear layout. The number of threads in a row or across positions does not really matter because we're going to create a linear order anyway.

The linear order gets created kind of like flat indexing and the flat indexing will start from 0, 0, until the 39th index I would say. It does not really have to end at a row boundary. Now Warp 0 ends at the end of the 32nd lane position and in this case the 32nd lane position is going to be T of 3, 1, and that means warp 1 should begin at T of 3, 2. That means the second warp (or let's say warp 1) will have lots of lane positions empty.

## 5. Two separate blocks

Each block has 16 threads. Both blocks are resident on the same SM.

- Can their threads combine into one full warp?
- Why, or why not?

My reasoning:

I don't think so because warps are supposed to be created by breaking down a block. If each block has 16 threads and we have two separate blocks, that means we will have two warps and both of them will not be fully filled. I don't think they are allowed to be combined, based on what I understand, because they depend on their blocks.

---

## Reference

The ordering rule follows PMPP Figure 4.7 and
[NVIDIA's thread-hierarchy documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#thread-hierarchy).
