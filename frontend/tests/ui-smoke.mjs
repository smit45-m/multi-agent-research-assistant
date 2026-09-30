import assert from "node:assert/strict";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { chromium } from "playwright-core";
import AxeBuilder from "@axe-core/playwright";

// Run against a local Vite server. APIs are intercepted only in this test browser.
// No real research, uploads, deletion, or benchmark jobs are performed.
const base = process.env.TEST_URL || "http://127.0.0.1:5173";
const output =
  process.env.TEST_OUTPUT || path.resolve(import.meta.dirname, "artifacts");
await mkdir(output, { recursive: true });
const browser = await chromium.launch({
  headless: true,
  ...(process.env.CHROMIUM_PATH
    ? { executablePath: process.env.CHROMIUM_PATH }
    : { channel: "chrome" }),
});
const results = [];
const consoleErrors = [];
const context = await browser.newContext({
  viewport: { width: 1440, height: 1100 },
  reducedMotion: "reduce",
});
const page = await context.newPage();
page.on("pageerror", (error) => consoleErrors.push(error.message));
let researchBehavior = "success";
let requests = 0;
let payload;
let benchmarkPayload;
let deleteFails = false;
let uploaded = false;
let releaseResearch;
const report = {
  task_id: "test-research-001",
  status: "completed",
  query: "Compare research approaches",
  report:
    '# A clear research report\n\nThis is a **test fixture**, not real research.\n\n## Key findings\n\n- Findings are linked to the returned evidence.\n- Trade-offs remain visible.\n\n<img src="x" onerror="window.__unsafe=true"><script>window.__unsafe=true</script>\n\n[Unsafe link](javascript:alert(1))',
  sources: [
    {
      title: "Example research source",
      url_or_path: "https://example.com/paper",
      relevance_score: 0.85,
      source_type: "web",
      snippet: "Test-only evidence excerpt.",
    },
  ],
  confidence_score: 0.87,
  response_accuracy_score: 0.88,
  synthesis_speedup_ratio: 0.6,
  processing_time_seconds: 3.21,
  orchestrator: "langgraph",
  created_at: "2026-09-14T10:00:00Z",
  telemetry: {
    planner_time_ms: 100,
    retriever_time_ms: 1100,
    analyzer_time_ms: 850,
    writer_time_ms: 1160,
    total_latency_ms: 3210,
  },
};
const document = {
  document_id: "test-document-001",
  filename: "context.txt",
  format: "txt",
  chunk_count: 3,
  status: "processed",
  uploaded_at: "2026-09-14T10:00:00Z",
};
await context.route("**/health/**", (route) =>
  route.fulfill({
    json: {
      status: "ok",
      version: "1.0.0",
      uptime_seconds: 1234,
      vector_store_documents: 3,
      active_concurrent_capacity: 50,
      latency_sla_seconds: 8,
      accuracy_benchmark_target: 85,
    },
  }),
);
await context.route("**/api/v1/research/sync", async (route) => {
  requests++;
  payload = route.request().postDataJSON();
  if (researchBehavior === "wait")
    await new Promise((resolve) => {
      releaseResearch = resolve;
    });
  if (researchBehavior === "failed")
    await route.fulfill({
      json: {
        ...report,
        status: "failed",
        report: "The configured model is unavailable.",
      },
    });
  else if (researchBehavior === "http")
    await route.fulfill({
      status: 503,
      json: { detail: "Research service temporarily unavailable." },
    });
  else await route.fulfill({ json: report });
});
await context.route("**/api/v1/documents/", (route) =>
  route.fulfill({
    json: {
      documents: uploaded ? [document] : [],
      total_count: uploaded ? 1 : 0,
    },
  }),
);
await context.route("**/api/v1/documents/upload", (route) => {
  uploaded = true;
  return route.fulfill({ json: document });
});
await context.route("**/api/v1/documents/test-document-001", (route) => {
  if (deleteFails)
    return route.fulfill({
      status: 500,
      json: { detail: "Could not remove library entry." },
    });
  uploaded = false;
  return route.fulfill({ json: { message: "Removed" } });
});
await context.route("**/api/v1/research/benchmark/run", (route) => {
  benchmarkPayload = route.request().postDataJSON();
  return route.fulfill({
    json: {
      total_test_cases: 2,
      passed_test_cases: 1,
      pass_rate_percentage: 50,
      average_accuracy_percentage: 75,
      target_accuracy_percentage: 85,
      average_latency_seconds: 4.2,
      target_latency_seconds: 8,
      average_synthesis_speedup_percentage: 60,
      target_synthesis_speedup_percentage: 60,
      categories_evaluated: ["Test category"],
      sample_results: [true, false].map((passed, index) => ({
        test_id: `test-${index}`,
        category: "Test category",
        query: `Test research question ${index}`,
        target_source_format: "web",
        accuracy_score: passed ? 90 : 60,
        latency_seconds: 4.2,
        synthesis_speedup: 60,
        passed,
        details: "",
      })),
    },
  });
});
async function test(name, fn) {
  await fn();
  results.push({ name, status: "passed" });
  console.log(`PASS ${name}`);
}
async function audit(name) {
  const analysis = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
    .analyze();
  const violations = analysis.violations.map((item) => ({
    id: item.id,
    impact: item.impact,
    nodes: item.nodes.map((node) => ({
      target: node.target,
      summary: node.failureSummary,
    })),
  }));
  await writeFile(
    path.join(output, `${name}-accessibility.json`),
    JSON.stringify(violations, null, 2),
  );
  assert.deepEqual(violations, [], `${name}: accessibility violations`);
}
async function noOverflow() {
  await page.evaluate(
    () =>
      new Promise((resolve) =>
        requestAnimationFrame(() => requestAnimationFrame(resolve)),
      ),
  );
  const dimensions = await page.evaluate(() => ({
    viewport: innerWidth,
    document: document.documentElement.scrollWidth,
    media: matchMedia("(max-width: 480px)").matches,
    bodyMargin: getComputedStyle(document.querySelector(".app-body"))
      .marginLeft,
    mainPadding: getComputedStyle(document.querySelector(".main-content"))
      .padding,
    htmlStyle: document.documentElement.getAttribute("style"),
    bodyStyle: document.body.getAttribute("style"),
    breadcrumb: document
      .querySelector(".breadcrumbs")
      .getBoundingClientRect()
      .toJSON(),
    overflowing: [...document.querySelectorAll("body *")]
      .filter(
        (element) => element.getBoundingClientRect().right > innerWidth + 1,
      )
      .map((element) => ({
        tag: element.tagName,
        class: element.className,
        right: element.getBoundingClientRect().right,
      }))
      .slice(0, 15),
  }));
  assert(
    dimensions.document <= dimensions.viewport + 1,
    `Unexpected horizontal overflow: ${JSON.stringify(dimensions)}`,
  );
}
try {
  await page.goto(base);
  await page
    .getByRole("button", { name: "Backend connected", exact: true })
    .waitFor();
  await test("Initial studio and desktop accessibility", async () => {
    assert(
      await page
        .getByRole("button", { name: "Start research", exact: true })
        .isDisabled(),
    );
    await noOverflow();
    await audit("desktop");
    await page.screenshot({
      path: path.join(output, "research-desktop.png"),
      fullPage: true,
    });
  });
  await test("Examples, manual settings, and keyboard submission", async () => {
    await page.getByRole("button", { name: /COMPARE & CONTRAST/ }).click();
    assert(
      (
        await page
          .getByRole("textbox", { name: "What would you like to understand?" })
          .inputValue()
      ).includes("fine-tuning"),
    );
    await page.getByText("Advanced settings", { exact: true }).click();
    await page
      .getByLabel("Retrieval mode", { exact: true })
      .selectOption("agentic");
    await page
      .getByRole("textbox", { name: "What would you like to understand?" })
      .press("Control+Enter");
    await page
      .getByRole("heading", { name: "A clear research report" })
      .waitFor();
    assert.equal(payload.rag_mode, "agentic");
    assert.equal(payload.orchestrator, "langgraph");
    assert.equal(requests, 1);
  });
  await test("Safe Markdown, sources, telemetry, and report download", async () => {
    assert.equal(await page.evaluate(() => !!window.__unsafe), false);
    assert.equal(
      await page
        .locator('.prose img, .prose script, .prose a[href^="javascript:"]')
        .count(),
      0,
    );
    const download = page.waitForEvent("download");
    await page.getByRole("button", { name: "Download", exact: true }).click();
    assert((await download).suggestedFilename().endsWith(".md"));
    await page.getByRole("tab", { name: "Sources (1)" }).click();
    assert.equal(
      await page
        .getByRole("link", { name: "Open original source" })
        .getAttribute("href"),
      "https://example.com/paper",
    );
    await page.getByRole("tab", { name: "Agent activity" }).click();
    assert(await page.getByText("1100 ms", { exact: true }).isVisible());
    await page.getByRole("tab", { name: "Research report" }).click();
    await audit("report");
    await page.screenshot({
      path: path.join(output, "research-report.png"),
      fullPage: true,
    });
  });
  await test("Research survives navigation; duplicate submits blocked", async () => {
    await page.getByRole("button", { name: "New research +" }).click();
    researchBehavior = "wait";
    await page
      .getByRole("textbox", { name: "What would you like to understand?" })
      .fill("A question for a slow research request");
    await page
      .getByRole("textbox", { name: "What would you like to understand?" })
      .press("Control+Enter");
    await page.getByText("Your research is underway").waitFor();
    await page
      .getByRole("textbox", { name: "What would you like to understand?" })
      .press("Control+Enter");
    await page
      .getByRole("button", { name: "System status", exact: true })
      .click();
    await page
      .getByRole("button", { name: "Research studio", exact: true })
      .click();
    assert(await page.getByText("Your research is underway").isVisible());
    assert.equal(requests, 2);
    await page.getByRole("button", { name: "Stop waiting" }).click();
    await page.getByText(/You stopped waiting/).waitFor();
    releaseResearch();
    researchBehavior = "success";
  });
  await test("HTTP 200 failed status and HTTP errors remain actionable", async () => {
    researchBehavior = "failed";
    await page
      .getByRole("button", { name: "Start research", exact: true })
      .click();
    await page.getByText("The configured model is unavailable.").waitFor();
    researchBehavior = "http";
    await page.getByRole("button", { name: "Try again", exact: true }).click();
    await page.getByText("Research service temporarily unavailable.").waitFor();
    researchBehavior = "success";
    await page.getByRole("button", { name: "Try again", exact: true }).click();
    await page
      .getByRole("heading", { name: "A clear research report" })
      .waitFor();
  });
  await test("Document upload, search, confirmation, and failed deletion", async () => {
    await page
      .getByRole("button", { name: "Knowledge library", exact: true })
      .click();
    await page
      .getByRole("heading", { name: "Make this library yours" })
      .waitFor();
    await page.locator('input[type="file"]').setInputFiles({
      name: "context.txt",
      mimeType: "text/plain",
      buffer: Buffer.from("Test-only context"),
    });
    await page
      .getByRole("button", { name: "Remove context.txt from library list" })
      .waitFor();
    await page
      .getByRole("textbox", { name: "Search documents" })
      .fill("no-match");
    assert(
      await page
        .getByRole("heading", { name: "No matching documents" })
        .isVisible(),
    );
    await page.getByRole("button", { name: "Clear search" }).click();
    await page
      .getByRole("button", { name: "Remove context.txt from library list" })
      .click();
    assert(await page.getByRole("dialog").isVisible());
    await audit("delete-dialog");
    await page.getByRole("button", { name: "Keep document" }).click();
    assert(uploaded);
    deleteFails = true;
    await page
      .getByRole("button", { name: "Remove context.txt from library list" })
      .click();
    await page
      .getByRole("button", { name: "Remove entry", exact: true })
      .click();
    await page
      .getByText("Could not remove library entry.", { exact: true })
      .waitFor();
    assert(uploaded);
    deleteFails = false;
    await page
      .getByRole("button", { name: "Remove context.txt from library list" })
      .click();
    await page
      .getByRole("button", { name: "Remove entry", exact: true })
      .click();
    await page
      .getByRole("heading", { name: "Make this library yours" })
      .waitFor();
    assert(!uploaded);
  });
  await test("Benchmark payload, honest failed cases, and filtering", async () => {
    await page.getByRole("button", { name: "Benchmarks", exact: true }).click();
    assert.equal(benchmarkPayload, undefined);
    await page.getByLabel("Number of test cases").selectOption("5");
    await page.getByRole("button", { name: "Run evaluation" }).click();
    await page.getByText("Failed", { exact: true }).waitFor();
    assert.equal(benchmarkPayload.max_cases, 5);
    await page
      .getByRole("textbox", { name: "Search benchmark cases" })
      .fill("test-1");
    assert.equal(await page.locator(".data-table tbody tr").count(), 1);
    await page
      .getByRole("textbox", { name: "Search benchmark cases" })
      .fill("");
    await audit("benchmarks");
  });
  await test("Theme persistence and light theme accessibility", async () => {
    await page
      .getByRole("button", { name: "Research studio", exact: true })
      .click();
    await page.getByRole("button", { name: "New research +" }).click();
    await page.getByRole("button", { name: "Switch to light theme" }).click();
    await page.reload();
    await page.getByRole("button", { name: "Switch to dark theme" }).waitFor();
    assert.equal(
      await page.locator("html").getAttribute("data-theme"),
      "light",
    );
    await audit("light");
    await noOverflow();
    await page.screenshot({
      path: path.join(output, "research-light.png"),
      fullPage: true,
    });
    await page.getByRole("button", { name: "Switch to dark theme" }).click();
  });
  await test("Mobile navigation, layout, and accessibility", async () => {
    await page.setViewportSize({ width: 390, height: 844 });
    await noOverflow();
    await audit("mobile");
    await page.screenshot({
      path: path.join(output, "research-mobile.png"),
      fullPage: true,
    });
    await page
      .getByRole("button", { name: "Open navigation", exact: true })
      .click();
    await page
      .getByRole("button", { name: "Knowledge library", exact: true })
      .click();
    await page
      .getByRole("heading", { name: "A little context goes a long way." })
      .waitFor();
    await noOverflow();
    await page
      .getByRole("button", { name: "Open navigation", exact: true })
      .click();
    await page.getByRole("button", { name: "Benchmarks", exact: true }).click();
    await noOverflow();
    await page
      .getByRole("button", { name: "Open navigation", exact: true })
      .click();
    await page
      .getByRole("button", { name: "Research studio", exact: true })
      .click();
    for (const width of [320, 768, 1024]) {
      await page.setViewportSize({ width, height: 1000 });
      await noOverflow();
    }
  });
  await test("No uncaught JavaScript errors", async () =>
    assert.deepEqual(consoleErrors, []));
  console.log(`${results.length} test groups passed.`);
} catch (error) {
  results.push({ name: "Failure", status: "failed", error: error.message });
  await page.screenshot({
    path: path.join(output, "test-failure.png"),
    fullPage: true,
  });
  console.error(error);
  process.exitCode = 1;
} finally {
  releaseResearch?.();
  await writeFile(
    path.join(output, "ui-test-results.json"),
    JSON.stringify(
      {
        note: "API responses are test fixtures; no live backend or model validation performed.",
        results,
        consoleErrors,
      },
      null,
      2,
    ),
  );
  await browser.close();
}
