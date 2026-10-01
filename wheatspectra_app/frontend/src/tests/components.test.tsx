import { describe, it, expect, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import "@testing-library/jest-dom";
import { ResponseBadge } from "../components/ResponseBadge";
import { PredictionCard } from "../components/PredictionCard";
import { Disclaimers } from "../components/Disclaimers";
import { MessageCard } from "../components/MessageCard";
import { ChatInterface } from "../components/ChatInterface";
import { EvidencePanel } from "../components/EvidencePanel";
import { ModelStatusModal } from "../components/ModelStatusModal";

describe("ResponseBadge Component", () => {
  it("renders scientific concept badge", () => {
    render(<ResponseBadge answerType="CONCEPT_EXPLANATION" />);
    expect(screen.getByText("Scientific concept")).toBeInTheDocument();
  });

  it("renders model prediction badge", () => {
    render(<ResponseBadge answerType="MODEL_PREDICTION" />);
    expect(screen.getByText("Model prediction")).toBeInTheDocument();
  });

  it("renders image required badge", () => {
    render(<ResponseBadge answerType="IMAGE_REQUIRED" />);
    expect(screen.getByText("Image required")).toBeInTheDocument();
  });

  it("renders runtime failure badge", () => {
    render(<ResponseBadge answerType="RUNTIME_ERROR" />);
    expect(screen.getByText("Runtime unavailable")).toBeInTheDocument();
  });
});

describe("PredictionCard Component", () => {
  it("renders predictions table with values and ensemble std", () => {
    const predictions = {
      Gsw: {
        value: 0.0579,
        unit: null,
        ensemble_std: 0.008,
        interval_low: null,
        interval_high: null,
        calibration_status: "NOT_AVAILABLE_FOR_LIVE_MODE",
      },
    };
    render(<PredictionCard predictions={predictions} />);
    expect(screen.getByText(/Gsw/i)).toBeInTheDocument();
    expect(screen.getByText(/0.0579/)).toBeInTheDocument();
    expect(screen.getByText(/± 0.0080/)).toBeInTheDocument();
  });
});

describe("Disclaimers Component", () => {
  it("renders counterfactual and spectral space disclaimers", () => {
    render(<Disclaimers />);
    expect(
      screen.getByText(/Evidence reflects model-internal counterfactual attribution/i)
    ).toBeInTheDocument();
  });
});

describe("MessageCard Component", () => {
  it("renders user message and copy button", () => {
    const userMsg = {
      id: "1",
      sender: "user" as const,
      text: "Estimate Gsw from this leaf sample.",
      timestamp: new Date().toISOString(),
    };
    render(<MessageCard message={userMsg} />);
    expect(screen.getByText("Estimate Gsw from this leaf sample.")).toBeInTheDocument();
    expect(screen.getByTitle("Copy message")).toBeInTheDocument();
  });

  it("renders assistant message with answer and copy button", () => {
    const assistantMsg = {
      id: "2",
      sender: "assistant" as const,
      response: {
        answer_mode: "VALIDATED" as const,
        answer_type: "MODEL_PREDICTION" as const,
        numeric_permission: true,
        intent: "QUANTITATIVE_ESTIMATION",
        selected_phenotypes: ["Gsw"],
        answer: "The estimated stomatal conductance (Gsw) is 0.0579.",
        predictions: {},
        visuals: {},
        verification: {
          all_dual_verified: true,
          evidence_status: "PASSED",
          response_policy: "VALIDATED_SCIENTIFIC_RESPONSE",
          downgrade_reasons: [],
        },
        evidence: {},
        warnings: [],
        scope: {
          hcsg: "PASSED",
          reason: "",
          open_set_score: 0,
          predicted_intent: "QUANTITATIVE_ESTIMATION",
        },
      },
      timestamp: new Date().toISOString(),
    };
    render(<MessageCard message={assistantMsg} />);
    expect(
      screen.getByText("The estimated stomatal conductance (Gsw) is 0.0579.")
    ).toBeInTheDocument();
    expect(screen.getByTitle("Copy answer")).toBeInTheDocument();
  });

  it("shows only predictions selected for the current answer", () => {
    const assistantMsg = {
      id: "selected-target-only",
      sender: "assistant" as const,
      response: {
        answer_mode: "EVIDENCE_BOUNDED" as const,
        answer_type: "MODEL_PREDICTION" as const,
        numeric_permission: true,
        intent: "TARGET_VALUE",
        selected_phenotypes: ["Cultivar"],
        answer: "The model predicts Cultivar = Sheriff.",
        predictions: {
          GrainWeight: {
            value: 8519.7871,
            unit: null,
            ensemble_std: null,
            interval_low: null,
            interval_high: null,
            calibration_status: "NOT_AVAILABLE_FOR_LIVE_MODE",
          },
          Cultivar: {
            value: "Sheriff",
            unit: null,
            ensemble_std: null,
            interval_low: null,
            interval_high: null,
            calibration_status: "NOT_AVAILABLE_FOR_LIVE_MODE",
          },
        },
        visuals: {},
        verification: {
          all_dual_verified: false,
          evidence_status: "VERIFIED_QWEN_OUTPUT",
          response_policy: "EVIDENCE_BOUNDED",
          downgrade_reasons: [],
        },
        evidence: {},
        warnings: [],
        scope: {
          hcsg: "SUPPORTED_PHENOTYPE",
          reason: "",
          open_set_score: 0,
          predicted_intent: "TARGET_VALUE",
        },
      },
      timestamp: new Date().toISOString(),
    };
    render(<MessageCard message={assistantMsg} />);
    expect(screen.getAllByText(/Cultivar/).length).toBeGreaterThan(0);
    expect(screen.queryByText("GrainWeight")).not.toBeInTheDocument();
    expect(screen.queryByText("8519.7871")).not.toBeInTheDocument();
  });
});

describe("ChatInterface knowledge chat", () => {
  it("sends a concept question without requiring an uploaded image", () => {
    const onSendMessage = vi.fn();
    render(
      <ChatInterface
        messages={[]}
        uploadData={null}
        activeFileName={null}
        inputPrompt="What is Cultivar?"
        isGenerating={false}
        isUploading={false}
        onInputChange={vi.fn()}
        onSendMessage={onSendMessage}
        onFileUpload={vi.fn()}
      />
    );

    fireEvent.click(screen.getByRole("button", { name: "Send message" }));
    expect(onSendMessage).toHaveBeenCalledWith("What is Cultivar?");
  });
});

describe("Target-specific visual explanation", () => {
  it("links a task prediction to its heatmap, focused crop, and text summary", () => {
    render(
      <EvidencePanel
        visuals={{
          target_visuals: {
            Cultivar: {
              target: "Cultivar",
              prediction_text: "Cultivar = Sheriff (ensemble vote 99%)",
              visual_method: "PCTG_COUNTERFACTUAL_PATCH_ATTRIBUTION" as const,
              evidence_available: true,
              focus_box: [90, 10, 134, 30],
              selected_token_indices: [50, 51, 81],
              selected_token_count: 3,
              spatial_summary: "The strongest connected evidence region is shown in the focused crop.",
              heatmap_url: "/media/cultivar_heatmap.png",
              marked_url: "/media/cultivar_marked.png",
              zoom_url: "/media/cultivar_zoom.png",
              bundle_url: "/media/cultivar_bundle.png",
            },
          },
        }}
        verification={{
          response_policy: "EVIDENCE_BOUNDED",
          evidence_status: "PARTIAL_OR_MISSING",
          all_dual_verified: false,
          downgrade_reasons: [],
        }}
        evidence={{}}
        warnings={[]}
      />
    );

    expect(screen.getByText("Cultivar = Sheriff (ensemble vote 99%)")).toBeInTheDocument();
    expect(screen.getByText("3 selected patches")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Target Heatmap" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Focused 4× Crop" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Text + Visual Bundle" })).toBeInTheDocument();
  });

  it("keeps verification visible when optional visual assets are missing", () => {
    render(
      <EvidencePanel
        visuals={{}}
        verification={{
          response_policy: "EVIDENCE_BOUNDED",
          evidence_status: "VERIFIED_GROUNDED_FALLBACK",
          all_dual_verified: false,
          downgrade_reasons: [],
          raw_answer_accepted: false,
          fallback_used: true,
          verification_reasons: ["raw_answer_contains_unauthorized_numbers"],
          ema_weights: true,
          vision_frozen: true,
        }}
        evidence={{}}
        warnings={[]}
      />
    );

    expect(screen.getByText("Visual assets unavailable")).toBeInTheDocument();
    expect(screen.getByText("Stage-33 Grounded Verifier")).toBeInTheDocument();
    expect(screen.getByText("VERIFIED_GROUNDED_FALLBACK")).toBeInTheDocument();
  });
});

describe("Dynamic model status", () => {
  it("renders the backend-reported ID-CV runtime instead of hard-coded 5-fold claims", () => {
    render(
      <ModelStatusModal
        isOpen
        onClose={vi.fn()}
        status={{
          cuda_available: true,
          device: "cuda",
          gpu_name: "NVIDIA GeForce RTX 4090",
          phytocore_loaded: true,
          phytocore_checkpoints_loaded: 15,
          qpr_loaded: true,
          hcsg_active: true,
          pctg_available: true,
          pgc_projector_loaded: true,
          qwen_loaded: true,
          lora_loaded: true,
          ver_available: true,
          scientific_ensemble_mode: "cv15",
          learned_parameters: { phytocore: 1_998_216 },
          total_learned_parameters: 1_998_216,
          supported_targets: ["Cultivar"],
          research_v2_vlm: {
            enabled: false,
            verified: false,
            loaded: false,
            scientific_result: false,
            activation_allowed: false,
            deployment_status: "DISABLED",
          },
          idcv_development_vlm: {
            enabled: true,
            verified: true,
            loaded: true,
            scientific_result: false,
            activation_allowed: true,
            deployment_status: "DEVELOPMENT_FOLD0",
            outer_fold: 0,
            model_seed: 2026,
            ema_weights: true,
            vision_frozen: true,
            architecture: "Frozen PhytoCore + visual bridge + Qwen-LoRA",
          },
        }}
      />
    );

    expect(screen.getByText("DEVELOPMENT_FOLD0")).toBeInTheDocument();
    expect(screen.getByText("Frozen PhytoCore + visual bridge + Qwen-LoRA")).toBeInTheDocument();
    expect(screen.getByText(/ID-CV DEVELOPMENT VLM/)).toBeInTheDocument();
    expect(screen.queryByText(/3 seeds × 5 folds/)).not.toBeInTheDocument();
  });
});
