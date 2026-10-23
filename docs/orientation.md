# What this project is

Read this before any code. No equations. Ten minutes.

## In one sentence

We are checking whether AI image processing quietly deletes small brain
lesions, while appearing to be *more* accurate than the traditional method.

## The background, in four steps

**1. A normal MRI scan is a picture, not a measurement.**
Brightness in an MRI is arbitrary. The same patient scanned on two machines
gives two different pictures. A radiologist reads the shapes, not the
numbers, because the numbers mean nothing on their own.

**2. Quantitative MRI tries to measure a real number instead.**
Instead of "this area looks brighter," it asks: what is the actual physical
property of this tissue? The property here is called **T1** — roughly, how
fast tissue recovers after you hit it with a magnetic pulse. Every tissue
has its own T1:

| tissue | T1 |
|---|---|
| white matter | about 832 ms |
| grey matter | about 1331 ms |
| fluid (CSF) | about 4000 ms |

These are physical constants. They should be the same on any machine, in any
country. That is the whole appeal: a number you can actually compare.

**3. You cannot measure T1 directly, so you work backwards.**
You take **five pictures** with slightly different scanner settings. Each
tissue produces a different *pattern* of five brightnesses. From the pattern,
you work backwards to the T1 that must have produced it.

The traditional method does this **one pixel at a time**, with maths. It is
slow and it ignores the fact that neighbouring pixels are usually the same
tissue.

**4. The new idea: let a neural network do it.**
A network is much faster, and it can look at *whole regions* instead of one
pixel. If one pixel's measurements are noisy, the network already knows what
the tissue around it looks like. It should win.

## The problem we are investigating

A network trained on normal brains learns what normal brains look like: big,
smooth, uniform regions.

Now show it a **tiny 3-pixel lesion**. It may quietly smooth it away —
because in everything it was ever trained on, tiny isolated spots were noise,
not real findings.

And here is the trap. **Its overall score will still look excellent**, because
98% of the image is ordinary tissue and it gets all of that right. The one
thing a doctor actually needed to see is the thing that got erased.

That is the project. Not "build a neural network for MRI" — but: *does the
network delete the thing we are looking for, and how would we even know?*

## The trick that makes it possible

We do not use real patient scans. We build a **fake brain** in the computer.

That sounds like a weakness. It is the entire point.

With a real scan, nobody knows the true T1 — that is exactly why this field
is hard. So if a method gives you 900 ms, you cannot tell whether it is right.

With a fake brain, **we chose** the true T1 for every pixel. So we can
simulate the five pictures, hand them to each method, and score every answer
against the truth we planted. That is the only way to catch a method that is
confidently wrong.

## The four methods we compare

| method | what it sees | what it tells us |
|---|---|---|
| traditional fit, no correction | one pixel | the naive baseline |
| traditional fit, corrected | one pixel | the real baseline to beat |
| MLP (simple network) | one pixel | does *learning* help? |
| U-Net (image network) | whole image | does *context* help — or hurt? |

The MLP is the clever part, and most people skip it. Without it, if the U-Net
wins you cannot say **why** — was it the learning, or was it the smoothing?
The MLP learns but cannot smooth, because it only ever sees one pixel. So it
separates the two.

## Where we are right now

Built and tested:

- the fake brain, and the physics that turns it into five pictures
- both traditional methods
- the MLP
- 105 automated tests that catch it if any of this breaks

Two findings so far:

**Finding 1.** The scanner's dial lies. When you ask for a 10-degree pulse,
different parts of the head actually receive 7 or 13 degrees. Ignoring this
makes white-matter T1 come out **11.5% too high**. Correcting it removes
almost all of the error. (This is the "B1" problem you will see in the code.)

**Finding 2.** The MLP beats the traditional method on every tissue it was
trained on — same accuracy, 12-27% less random scatter. But on **lesions**,
which it never saw in training, it is off by +25 ms where the traditional
method is off by +0.1 ms.

Scored over the whole image, the MLP looks 25% better. Scored on lesions
alone, it is 21% worse. Lesions are 2% of the image, so the overall score
hides the failure completely.

**That is your thesis, already demonstrated** — and we have not even built
the U-Net yet.

## What is left

- a way to measure "did the lesion survive?" (in progress)
- the U-Net
- repeat everything with several random seeds, so it is a result and not a fluke
- write it up

## How much trouble are you in?

Less than it feels. The original plan said the core project was: simulator,
both traditional methods, MLP, U-Net, and per-tissue error. **Four of those
five are done and tested.** What is left is the U-Net and the write-up.

The thing that matters now is not more code. It is that you can explain this
page in your own words. If you can, you can defend the project. If you
cannot, no amount of extra code helps.
