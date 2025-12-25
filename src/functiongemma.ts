import {
  AutoTokenizer,
  AutoModelForCausalLM,
  type PreTrainedTokenizer,
  type PreTrainedModel,
  type Tensor,
} from "@huggingface/transformers";

const MODEL_ID = "onnx-community/functiongemma-270m-it-ONNX";

// Model size information (approximate, based on fp16)
export const MODEL_INFO = {
  name: "FunctionGemma-270M",
  modelId: MODEL_ID,
  tokenizerSize: 2.5 * 1024 * 1024, // ~2.5 MB
  modelSize: 540 * 1024 * 1024, // ~540 MB (fp16)
  get totalSize() {
    return this.tokenizerSize + this.modelSize;
  },
  get totalSizeMB() {
    return Math.round(this.totalSize / (1024 * 1024));
  },
};

let tokenizer: PreTrainedTokenizer | null = null;
let model: PreTrainedModel | null = null;

export interface FunctionSchema {
  type: "function";
  function: {
    name: string;
    description: string;
    parameters: {
      type: "object";
      properties: Record<
        string,
        {
          type: string;
          description: string;
        }
      >;
      required: string[];
    };
  };
}

export const counterFunctionSchemas: FunctionSchema[] = [
  {
    type: "function",
    function: {
      name: "increment",
      description: "Add 1 to the counter",
      parameters: {
        type: "object",
        properties: {},
        required: [],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "decrement",
      description: "Subtract 1 from the counter",
      parameters: {
        type: "object",
        properties: {},
        required: [],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "set_counter",
      description: "Set the counter to a specific number",
      parameters: {
        type: "object",
        properties: {
          number: {
            type: "string",
            description: "The number to set",
          },
        },
        required: ["number"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "reset_counter",
      description: "Reset the counter to 0",
      parameters: {
        type: "object",
        properties: {},
        required: [],
      },
    },
  },
];

export interface LoadingProgress {
  status: string;
  progress?: number;
  file?: string;
  loaded?: number;
  total?: number;
}

export async function loadModel(
  onProgress?: (progress: LoadingProgress) => void
): Promise<void> {
  if (tokenizer && model) {
    return;
  }

  onProgress?.({
    status: "loading",
    file: "tokenizer",
    loaded: 0,
    total: MODEL_INFO.tokenizerSize,
  });

  tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID, {
    progress_callback: (p) => {
      if (p && typeof p === "object" && "progress" in p) {
        const progress = p.progress as number | undefined;
        const loaded = (p as { loaded?: number }).loaded;
        const total = (p as { total?: number }).total;
        if (progress !== undefined) {
          onProgress?.({
            status: "loading",
            file: "tokenizer",
            progress,
            loaded: loaded ?? Math.round((progress / 100) * MODEL_INFO.tokenizerSize),
            total: total ?? MODEL_INFO.tokenizerSize,
          });
        }
      }
    },
  });

  onProgress?.({
    status: "loading",
    file: "model",
    loaded: 0,
    total: MODEL_INFO.modelSize,
  });

  model = await AutoModelForCausalLM.from_pretrained(MODEL_ID, {
    dtype: "fp16",
    device: "webgpu",
    progress_callback: (p) => {
      if (p && typeof p === "object" && "progress" in p) {
        const progress = p.progress as number | undefined;
        const loaded = (p as { loaded?: number }).loaded;
        const total = (p as { total?: number }).total;
        if (progress !== undefined) {
          onProgress?.({
            status: "loading",
            file: "model",
            progress,
            loaded: loaded ?? Math.round((progress / 100) * MODEL_INFO.modelSize),
            total: total ?? MODEL_INFO.modelSize,
          });
        }
      }
    },
  });

  onProgress?.({ status: "ready" });
}

// ============================================
// Preprocessing: Multilingual to English translation
// ============================================
const MULTILINGUAL_TO_ENGLISH: { pattern: RegExp; replacement: string }[] = [
  // Special number patterns (must be first)
  { pattern: /\+1/g, replacement: "increment" },
  { pattern: /-1/g, replacement: "decrement" },
  { pattern: /\bup by one\b/gi, replacement: "increment" },
  { pattern: /\bdown by one\b/gi, replacement: "decrement" },
  { pattern: /\bplus 1\b/gi, replacement: "increment" },
  { pattern: /\bminus 1\b/gi, replacement: "decrement" },

  // Set patterns with numbers
  { pattern: /make it (\d+)/gi, replacement: "set to $1" },
  { pattern: /make the counter (\d+)/gi, replacement: "set counter to $1" },

  // Compound phrases (before individual words)
  { pattern: /go down/gi, replacement: "decrement" },
  { pattern: /go up/gi, replacement: "increment" },
  { pattern: /count up/gi, replacement: "increment" },
  { pattern: /count down/gi, replacement: "decrement" },
  { pattern: /go lower/gi, replacement: "decrement" },
  { pattern: /go higher/gi, replacement: "increment" },
  { pattern: /하나 더/g, replacement: "add one" },
  { pattern: /하나 빼/g, replacement: "subtract one" },
  { pattern: /one more/gi, replacement: "increment" },
  { pattern: /one less/gi, replacement: "decrement" },
  { pattern: /tick up/gi, replacement: "increment" },
  { pattern: /tick down/gi, replacement: "decrement" },
  { pattern: /bump it up/gi, replacement: "increment" },
  { pattern: /bump up/gi, replacement: "increment" },
  { pattern: /raise it/gi, replacement: "increment" },
  { pattern: /lower it/gi, replacement: "decrement" },
  { pattern: /lower the counter/gi, replacement: "decrement" },
  { pattern: /raise the counter/gi, replacement: "increment" },
  { pattern: /reduce it/gi, replacement: "decrement" },
  { pattern: /back to zero/gi, replacement: "reset" },
  { pattern: /go to zero/gi, replacement: "reset" },
  { pattern: /start over/gi, replacement: "reset" },
  { pattern: /set to zero/gi, replacement: "reset" },
  { pattern: /make it zero/gi, replacement: "reset" },
  { pattern: /zero out/gi, replacement: "reset" },
  { pattern: /take one away/gi, replacement: "decrement" },
  { pattern: /add one more/gi, replacement: "increment" },
  { pattern: /count one more/gi, replacement: "increment" },
  { pattern: /count one less/gi, replacement: "decrement" },
  { pattern: /make it higher/gi, replacement: "increment" },
  { pattern: /make it lower/gi, replacement: "decrement" },
  { pattern: /add to counter/gi, replacement: "increment" },
  { pattern: /subtract from counter/gi, replacement: "decrement" },
  { pattern: /counter up/gi, replacement: "increment" },
  { pattern: /counter down/gi, replacement: "decrement" },

  // ============================================
  // Japanese keywords (日本語)
  // ============================================

  // Japanese compound phrases (must be before single words)
  { pattern: /一つ増やし?て?/g, replacement: "increment" },
  { pattern: /ひとつ増やし?て?/g, replacement: "increment" },
  { pattern: /1増やし?て?/g, replacement: "increment" },
  { pattern: /一つ減らし?て?/g, replacement: "decrement" },
  { pattern: /ひとつ減らし?て?/g, replacement: "decrement" },
  { pattern: /1減らし?て?/g, replacement: "decrement" },
  { pattern: /一つ足し?て?/g, replacement: "increment" },
  { pattern: /ひとつ足し?て?/g, replacement: "increment" },
  { pattern: /一つ引い?て?/g, replacement: "decrement" },
  { pattern: /ひとつ引い?て?/g, replacement: "decrement" },
  { pattern: /1つ減らせ/g, replacement: "decrement" },

  // Japanese "another one" patterns
  { pattern: /もう一つ/g, replacement: "increment" },
  { pattern: /もうひとつ/g, replacement: "increment" },
  { pattern: /もう1つ/g, replacement: "increment" },
  { pattern: /あと一つ/g, replacement: "increment" },
  { pattern: /あと1つ/g, replacement: "increment" },

  // Japanese katakana loanwords
  { pattern: /インクリーズ/g, replacement: "increment" },
  { pattern: /アドワン/g, replacement: "increment" },
  { pattern: /アド/g, replacement: "increment" },
  { pattern: /ディクリーズ/g, replacement: "decrement" },
  { pattern: /サブトラクト/g, replacement: "decrement" },
  { pattern: /マイナス1/g, replacement: "decrement" },
  { pattern: /マイナス一/g, replacement: "decrement" },

  // Japanese compound words (before individual)
  { pattern: /値引き/g, replacement: "decrement" },

  // Japanese question/request forms (must be before suffix removal)
  { pattern: /増やしてくれる\?/g, replacement: "increment" },
  { pattern: /増やしてもらえる\?/g, replacement: "increment" },
  { pattern: /増やせる\?/g, replacement: "increment" },
  { pattern: /足してくれる\?/g, replacement: "increment" },
  { pattern: /足してもらえる\?/g, replacement: "increment" },
  { pattern: /減らしてくれる\?/g, replacement: "decrement" },
  { pattern: /減らしてもらえる\?/g, replacement: "decrement" },
  { pattern: /減らせる\?/g, replacement: "decrement" },
  { pattern: /引いてくれる\?/g, replacement: "decrement" },
  { pattern: /下げてくれる\?/g, replacement: "decrement" },

  // Japanese desire/want forms with suffixes (〜したいんだけど, etc.)
  { pattern: /増やしたいんだけど/g, replacement: "increment" },
  { pattern: /増やしたいな/g, replacement: "increment" },
  { pattern: /増やしたい/g, replacement: "increment" },
  { pattern: /足したいんだけど/g, replacement: "increment" },
  { pattern: /足したい/g, replacement: "increment" },
  { pattern: /上げたいんだけど/g, replacement: "increment" },
  { pattern: /上げたいな/g, replacement: "increment" },
  { pattern: /上げたい/g, replacement: "increment" },
  { pattern: /減らしたいんだけど/g, replacement: "decrement" },
  { pattern: /減らしたい/g, replacement: "decrement" },
  { pattern: /引きたい/g, replacement: "decrement" },
  { pattern: /下げたいんだけど/g, replacement: "decrement" },
  { pattern: /下げたいな/g, replacement: "decrement" },
  { pattern: /下げたい/g, replacement: "decrement" },

  // Japanese imperative short forms
  { pattern: /増やせ/g, replacement: "increment" },
  { pattern: /足せ/g, replacement: "increment" },
  { pattern: /減らせ/g, replacement: "decrement" },
  { pattern: /引け/g, replacement: "decrement" },
  { pattern: /引き/g, replacement: "decrement" },
  { pattern: /ゼロに戻す/g, replacement: "reset" },
  { pattern: /ゼロに戻して/g, replacement: "reset" },
  { pattern: /ゼロに戻し/g, replacement: "reset" },
  { pattern: /0に戻す/g, replacement: "reset" },
  { pattern: /0に戻して/g, replacement: "reset" },
  { pattern: /0に戻し/g, replacement: "reset" },
  { pattern: /元に戻す/g, replacement: "reset" },
  { pattern: /元に戻して/g, replacement: "reset" },
  { pattern: /元に戻し/g, replacement: "reset" },
  { pattern: /初期値に戻す/g, replacement: "reset" },
  { pattern: /初期値に戻して/g, replacement: "reset" },
  { pattern: /初期値に/g, replacement: "reset" },
  { pattern: /ゼロにする/g, replacement: "reset" },
  { pattern: /ゼロにして/g, replacement: "reset" },
  { pattern: /0にリセット/g, replacement: "reset" },

  // Japanese increment keywords (all verb conjugations)
  { pattern: /増加/g, replacement: "increment" },
  { pattern: /増やす/g, replacement: "increment" },
  { pattern: /増やして/g, replacement: "increment" },
  { pattern: /増やし/g, replacement: "increment" },
  { pattern: /プラス/g, replacement: "increment" },
  { pattern: /足す/g, replacement: "increment" },
  { pattern: /足して/g, replacement: "increment" },
  { pattern: /足し/g, replacement: "increment" },
  { pattern: /上げる/g, replacement: "increment" },
  { pattern: /上げて/g, replacement: "increment" },
  { pattern: /上げ/g, replacement: "increment" },
  { pattern: /アップ/g, replacement: "increment" },
  { pattern: /インクリメント/g, replacement: "increment" },
  { pattern: /加算/g, replacement: "increment" },
  { pattern: /加える/g, replacement: "increment" },
  { pattern: /加えて/g, replacement: "increment" },

  // Japanese decrement keywords (all verb conjugations)
  { pattern: /減少/g, replacement: "decrement" },
  { pattern: /減らす/g, replacement: "decrement" },
  { pattern: /減らして/g, replacement: "decrement" },
  { pattern: /減らし/g, replacement: "decrement" },
  { pattern: /マイナス/g, replacement: "decrement" },
  { pattern: /引く/g, replacement: "decrement" },
  { pattern: /引いて/g, replacement: "decrement" },
  { pattern: /引い/g, replacement: "decrement" },
  { pattern: /下げる/g, replacement: "decrement" },
  { pattern: /下げて/g, replacement: "decrement" },
  { pattern: /下げ/g, replacement: "decrement" },
  { pattern: /ダウン/g, replacement: "decrement" },
  { pattern: /デクリメント/g, replacement: "decrement" },
  { pattern: /減算/g, replacement: "decrement" },

  // Japanese set keywords
  { pattern: /設定/g, replacement: "set" },
  { pattern: /セット/g, replacement: "set" },
  { pattern: /変更/g, replacement: "change" },
  { pattern: /変えて?/g, replacement: "change" },
  { pattern: /(\d+)に設定/g, replacement: "set to $1" },
  { pattern: /(\d+)にセット/g, replacement: "set to $1" },
  { pattern: /(\d+)にして?/g, replacement: "set to $1" },
  { pattern: /(\d+)に変更/g, replacement: "change to $1" },
  { pattern: /(\d+)に変えて?/g, replacement: "change to $1" },

  // Japanese reset keywords
  { pattern: /リセット/g, replacement: "reset" },
  { pattern: /初期化/g, replacement: "reset" },
  { pattern: /クリア/g, replacement: "reset" },
  { pattern: /ゼロ/g, replacement: "zero" },
  { pattern: /初期値/g, replacement: "reset" },
  { pattern: /初めから/g, replacement: "reset" },
  { pattern: /最初から/g, replacement: "reset" },
  { pattern: /最初に戻/g, replacement: "reset" },
  { pattern: /やり直/g, replacement: "reset" },
  { pattern: /取り消/g, replacement: "reset" },
  { pattern: /なかったことに/g, replacement: "reset" },
  { pattern: /白紙に戻/g, replacement: "reset" },
  { pattern: /消して/g, replacement: "reset" },
  { pattern: /消し/g, replacement: "reset" },
  { pattern: /全部消/g, replacement: "reset" },
  { pattern: /0にし/g, replacement: "set to 0" },

  // Japanese set with context (カウンターを{num}に)
  { pattern: /を(\d+)に$/g, replacement: " set to $1" },

  // Japanese suffixes (remove polite/casual endings)
  { pattern: /をお願いします$/g, replacement: "" },
  { pattern: /ようお願いします$/g, replacement: "" },
  { pattern: /してください$/g, replacement: "" },
  { pattern: /してくれる\?$/g, replacement: "" },
  { pattern: /してくれない\?$/g, replacement: "" },
  { pattern: /してもらえる\?$/g, replacement: "" },
  { pattern: /できる\?$/g, replacement: "" },
  { pattern: /してくれ$/g, replacement: "" },
  { pattern: /してほしい$/g, replacement: "" },
  { pattern: /してほしいな$/g, replacement: "" },
  { pattern: /したい$/g, replacement: "" },
  { pattern: /したいな$/g, replacement: "" },
  { pattern: /したいんだけど$/g, replacement: "" },
  { pattern: /しなさい$/g, replacement: "" },
  { pattern: /するんだ$/g, replacement: "" },
  { pattern: /せる\?$/g, replacement: "" },
  { pattern: /して$/g, replacement: "" },
  { pattern: /ください$/g, replacement: "" },
  { pattern: /くれ$/g, replacement: "" },
  { pattern: /しろ$/g, replacement: "" },
  { pattern: /せよ$/g, replacement: "" },
  { pattern: /よ$/g, replacement: "" },
  { pattern: /な$/g, replacement: "" },
  { pattern: /する$/g, replacement: "" },
  { pattern: /て$/g, replacement: "" },

  // Japanese context words (add trailing space for proper word separation)
  { pattern: /カウンター?を?/g, replacement: "counter " },
  { pattern: /カウンタを?/g, replacement: "counter " },
  { pattern: /数字を?/g, replacement: "number " },
  { pattern: /数を?/g, replacement: "value " },
  { pattern: /値を?/g, replacement: "value " },

  // ============================================
  // Korean keywords (한국어)
  // ============================================

  // Increment keywords (Korean)
  { pattern: /증가/g, replacement: "increment" },
  { pattern: /올려/g, replacement: "increment" },
  { pattern: /더해/g, replacement: "increment" },
  { pattern: /플러스/g, replacement: "increment" },
  { pattern: /추가/g, replacement: "increment" },
  { pattern: /높여/g, replacement: "increment" },
  { pattern: /키워/g, replacement: "increment" },
  { pattern: /늘려/g, replacement: "increment" },
  { pattern: /인크리먼트/g, replacement: "increment" },
  { pattern: /업/g, replacement: "increment" },

  // Decrement keywords (Korean)
  { pattern: /감소/g, replacement: "decrement" },
  { pattern: /내려/g, replacement: "decrement" },
  { pattern: /빼/g, replacement: "decrement" },
  { pattern: /마이너스/g, replacement: "decrement" },
  { pattern: /줄여/g, replacement: "decrement" },
  { pattern: /낮춰/g, replacement: "decrement" },
  { pattern: /작게/g, replacement: "decrement" },
  { pattern: /디크리먼트/g, replacement: "decrement" },
  { pattern: /다운/g, replacement: "decrement" },

  // Set keywords (Korean)
  { pattern: /설정/g, replacement: "set" },
  { pattern: /세팅/g, replacement: "set" },
  { pattern: /변경/g, replacement: "change" },
  { pattern: /바꿔/g, replacement: "change" },
  { pattern: /(\d+)으로/g, replacement: "to $1" },
  { pattern: /(\d+)로/g, replacement: "to $1" },

  // Reset keywords (Korean)
  { pattern: /초기화/g, replacement: "reset" },
  { pattern: /리셋/g, replacement: "reset" },
  { pattern: /클리어/g, replacement: "clear" },
  { pattern: /제로/g, replacement: "zero" },
  { pattern: /처음으로/g, replacement: "reset" },
  { pattern: /원래대로/g, replacement: "reset" },
  { pattern: /기본값/g, replacement: "reset" },
  { pattern: /0으로/g, replacement: "to 0" },

  // English keywords that need normalization
  { pattern: /\bincrease\b/gi, replacement: "increment" },
  { pattern: /\badd\b/gi, replacement: "increment" },
  { pattern: /\bplus\b/gi, replacement: "increment" },
  { pattern: /\bdecrease\b/gi, replacement: "decrement" },
  { pattern: /\bsubtract\b/gi, replacement: "decrement" },
  { pattern: /\bminus\b/gi, replacement: "decrement" },
  { pattern: /\braise\b/gi, replacement: "increment" },
  { pattern: /\breduce\b/gi, replacement: "decrement" },
  { pattern: /\bclear\b/gi, replacement: "reset" },
  { pattern: /\bdefault\b/gi, replacement: "reset" },
  { pattern: /\bdown\b/gi, replacement: "decrement" },
  { pattern: /\bup\b/gi, replacement: "increment" },

  // Common suffixes (Korean)
  { pattern: /해줘$/g, replacement: "" },
  { pattern: /시켜줘$/g, replacement: "" },
  { pattern: /시켜$/g, replacement: "" },
  { pattern: /해$/g, replacement: "" },
  { pattern: /줘$/g, replacement: "" },

  // Context words (add trailing space for proper word separation)
  { pattern: /카운터를?/g, replacement: "counter " },
  { pattern: /숫자를?/g, replacement: "number " },
  { pattern: /값을?/g, replacement: "value " },
  { pattern: /하나/g, replacement: "one" },
];

function preprocessInput(input: string): string {
  let result = input;
  for (const { pattern, replacement } of MULTILINGUAL_TO_ENGLISH) {
    result = result.replace(pattern, replacement);
  }
  // Clean up multiple spaces
  result = result.replace(/\s+/g, " ").trim();
  return result;
}

// ============================================
// Postprocessing: Function name normalization
// ============================================
const FUNCTION_NAME_MAP: Record<string, string> = {
  // Increment variations
  increment: "increment",
  INCREMENT: "increment",
  Increment: "increment",
  add: "increment",
  ADD: "increment",
  plus: "increment",
  PLUS: "increment",
  increase: "increment",
  INCREASE: "increment",

  // Decrement variations
  decrement: "decrement",
  DECREMENT: "decrement",
  Decrement: "decrement",
  decrease: "decrement",
  DECREASE: "decrement",
  Decrease: "decrement",
  subtract: "decrement",
  SUBTRACT: "decrement",
  minus: "decrement",
  MINUS: "decrement",

  // Set counter variations
  set_counter: "set_counter",
  SET_COUNTER: "set_counter",
  set: "set_counter",
  SET: "set_counter",
  print_counter: "set_counter",

  // Reset counter variations
  reset_counter: "reset_counter",
  RESET_COUNTER: "reset_counter",
  reset: "reset_counter",
  RESET: "reset_counter",
  clear_counter: "reset_counter",
  CLEAR_COUNTER: "reset_counter",
  clear: "reset_counter",
  CLEAR: "reset_counter",
};

function normalizeFunction(name: string): string {
  return FUNCTION_NAME_MAP[name] || name.toLowerCase();
}

export interface FunctionCall {
  name: string;
  args: Record<string, unknown>;
}

function parseFunctionCall(
  output: string,
  processedInput: string
): FunctionCall | null {
  const match = output.match(/call:(\w+)\{([^}]*)\}/);
  if (match) {
    const rawName = match[1];
    let name = normalizeFunction(rawName);
    const argsString = match[2];
    const args: Record<string, unknown> = {};

    if (argsString) {
      const argMatches = argsString.matchAll(/(\w+):<escape>([^<]*)<escape>/g);
      for (const m of argMatches) {
        args[m[1]] = m[2];
      }
      // Also try without escape tags
      if (Object.keys(args).length === 0) {
        const simpleMatches = argsString.matchAll(/(\w+):([^,}]+)/g);
        for (const m of simpleMatches) {
          args[m[1]] = m[2];
        }
      }
    }

    // Post-processing: if set_counter with +1 or -1, convert to increment/decrement
    // If set_counter with 0, convert to reset_counter
    if (name === "set_counter" && args.number !== undefined) {
      const numStr = String(args.number).trim();
      if (numStr === "+1") {
        name = "increment";
        delete args.number;
      } else if (numStr === "-1") {
        name = "decrement";
        delete args.number;
      } else if (numStr === "0") {
        name = "reset_counter";
        delete args.number;
      }
    }

    // Context-aware correction based on processed input
    const inputLower = processedInput.toLowerCase();
    const hasDecrementKeyword =
      /\b(decrement|subtract|minus|decrease)\b/.test(inputLower);
    const hasIncrementKeyword =
      /\b(increment|add|plus|increase)\b/.test(inputLower);
    const hasResetKeyword = /\b(reset|clear|zero)\b/.test(inputLower);
    const hasSetPattern =
      /\b(set|change)\b.*\b\d+\b|\bto\s+\d+\b/.test(inputLower);

    // Check if resetting to zero (has reset keyword and involves 0)
    const isSettingToZero =
      /\bto\s+0\b|\b0\b/.test(inputLower) && hasResetKeyword;

    // If input has reset keyword with 0, prefer reset_counter
    if (
      isSettingToZero &&
      (name === "set_counter" || name === "reset_counter")
    ) {
      name = "reset_counter";
      delete args.number;
    }
    // If input has explicit "set to N" or "change to N" pattern (non-zero), prefer set_counter
    else if (hasSetPattern && name === "reset_counter" && !isSettingToZero) {
      name = "set_counter";
      const numMatch = inputLower.match(/\bto\s+(\d+)\b|\b(\d+)\b/);
      if (numMatch) {
        args.number = numMatch[1] || numMatch[2];
      }
    }
    // Only override if input has a clear single intent and no set pattern
    else if (
      hasDecrementKeyword &&
      !hasIncrementKeyword &&
      !hasResetKeyword &&
      !hasSetPattern
    ) {
      if (name === "set_counter" || name === "reset_counter") {
        name = "decrement";
        delete args.number;
      }
    } else if (
      hasIncrementKeyword &&
      !hasDecrementKeyword &&
      !hasResetKeyword &&
      !hasSetPattern
    ) {
      if (name === "set_counter" || name === "reset_counter") {
        name = "increment";
        delete args.number;
      }
    } else if (
      hasResetKeyword &&
      !hasDecrementKeyword &&
      !hasIncrementKeyword &&
      !hasSetPattern
    ) {
      if (
        name === "set_counter" ||
        name === "increment" ||
        name === "decrement"
      ) {
        name = "reset_counter";
        delete args.number;
      }
    }

    return { name, args };
  }

  return null;
}

// Optimized prompt for function calling
const OPTIMIZED_PROMPT = `You are a function calling assistant. Always respond with a function call.

Functions:
- increment: Add 1
- decrement: Subtract 1
- set_counter: Set to number (parameter: number)
- reset_counter: Reset to 0

Rules:
- "increase", "add", "plus", "up" → increment
- "decrease", "subtract", "minus", "down" → decrement
- "set to N", "change to N" → set_counter with number=N
- "reset", "clear", "zero" → reset_counter

IMPORTANT: Always use the exact function names above.`;

export async function generateFunctionCall(
  userInput: string
): Promise<FunctionCall | null> {
  if (!tokenizer || !model) {
    throw new Error("Model not loaded. Call loadModel() first.");
  }

  // Pre-process input (Korean → English)
  const processedInput = preprocessInput(userInput);

  const messages = [
    {
      role: "developer",
      content: OPTIMIZED_PROMPT,
    },
    {
      role: "user",
      content: processedInput,
    },
  ];

  const inputs = tokenizer.apply_chat_template(messages, {
    tools: counterFunctionSchemas,
    tokenize: true,
    add_generation_prompt: true,
    return_dict: true,
  }) as { input_ids: Tensor; attention_mask: Tensor };

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const output = (await (model as any).generate({
    ...inputs,
    max_new_tokens: 128,
    do_sample: false,
  })) as Tensor;

  const inputLength = inputs.input_ids.dims[1];
  const outputData = Array.from(output.data as BigInt64Array);
  const generatedTokens = outputData.slice(inputLength);

  const decoded = tokenizer.decode(generatedTokens, {
    skip_special_tokens: false,
  });

  console.log("=== FunctionGemma Debug ===");
  console.log("Original input:", userInput);
  console.log("Processed input:", processedInput);
  console.log("Raw output:", decoded);
  console.log("===========================");

  return parseFunctionCall(decoded, processedInput);
}

export function isModelLoaded(): boolean {
  return tokenizer !== null && model !== null;
}
