
import { useState } from "react";
import { PatientForm } from "@/components/PatientForm";
import { RecommendationResults } from "@/components/RecommendationResults";
import { Activity } from "lucide-react";
import {
  predictDrugs,
  type PatientInput,
  type PredictionResponse,
} from "@/lib/api-client";

const Index = () => {
  const [results, setResults] = useState<PredictionResponse | null>(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handlePredict = async (patient: PatientInput) => {
    setIsPredicting(true);
    setError(null);

    try {
      const prediction = await predictDrugs(patient);
      setResults(prediction);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to get predictions"
      );
    } finally {
      setIsPredicting(false);
    }
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b bg-card">
        <div className="container mx-auto px-4 py-6">
          <div className="flex items-center gap-3">
            <Activity className="h-8 w-8 text-primary" />
            <div>
              <h1 className="text-2xl font-bold text-foreground">
                Drug Recommendation System
              </h1>
              <p className="text-sm text-muted-foreground">
                AI-powered medication suggestions with SHAP explanations
              </p>
            </div>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8 grid gap-8 lg:grid-cols-2">
        {/* Patient Input */}
        <section>
          <h2 className="text-lg font-semibold mb-3">
            Patient Information
          </h2>
          <PatientForm onSubmit={handlePredict} isLoading={isPredicting} />
        </section>

        {/* Results */}
        <section>
          <h2 className="text-lg font-semibold mb-3">
            Recommendations
          </h2>

          {error && (
            <p className="text-sm text-red-500 mb-4">{error}</p>
          )}

          {results ? (
            <RecommendationResults results={results} />
          ) : (
            <div className="flex items-center justify-center h-96 border-2 border-dashed rounded-lg">
              <p className="text-muted-foreground text-center">
                Enter patient details to generate drug recommendations
              </p>
            </div>
          )}
        </section>
      </main>
    </div>
  );
};

export default Index;
