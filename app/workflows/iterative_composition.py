from app.agents.Andy import Andy
from app.objects.iterative_composition_workflow import IterativeCompositionWorkflow
from app.objects.rainbow_color import RainbowColor

if __name__ == "__main__":
    print("🎵 Testing Iterative Composition Workflow")
    print("=" * 50)

    andy = Andy()
    andy.initialize()

    # Create workflow
    workflow = IterativeCompositionWorkflow(andy)

    # Execute complete workflow
    result = workflow.execute_workflow(RainbowColor.Z)

    print("\n" + "=" * 50)
    print("🎯 FINAL COMPOSITION RESULTS")
    print("=" * 50)

    print(f"Session ID: {result['session_id']}")
    print(f"Overall Quality Score: {result['quality_score']:.2f}")

    print(f"\nIterations Used:")
    for stage, count in result['iterations_used'].items():
        if count > 0:
            print(f"  {stage}: {count}")

    print(f"\nFinal Approvals:")
    for feedback in result['feedback_summary']:
        print(f"  {feedback['stage']}: {'✅' if feedback['final_approval'] else '❌'} ({feedback['confidence']:.2f})")

    print(f"\nGenerated Plan:")
    plan = result['plan']
    if plan:
        print(f"  Key: {plan.get('key', 'Unknown')}")
        print(f"  Tempo: {plan.get('tempo', 'Unknown')} BPM")
        print(f"  Sections: {plan.get('sections', 'Unknown')}")

    if result['chord_progressions']:
        print(f"\nChord Progressions Generated: {len(result['chord_progressions'])}")

    if result['lyrics']:
        print(f"\nLyrics Sections: {len(result['lyrics'].get('sections', []))}")

    print("\n🎉 Workflow Complete!")
