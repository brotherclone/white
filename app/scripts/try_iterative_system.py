import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.agents.Andy import Andy
from app.objects.rainbow_color import RainbowColor


def try_basic_system():
    """Test the basic system as it exists now"""
    print("🎵 Testing Current System")
    print("=" * 40)

    try:
        # Initialize Andy
        andy = Andy()
        andy.initialize()

        # Generate a basic plan
        print("🎯 Generating plan...")
        plan = andy.planning_agent.generate_plans(rainbow_color=RainbowColor.Z)

        print("Plan generated:")
        for key, value in plan.items():
            print(f"  {key}: {value}")

        # Convert to song plan
        print("\n🎼 Converting to song structure...")
        song_plan = andy.planning_agent.env.plan_to_rainbow_song_plan(plan)

        print(f"Song Plan:")
        print(f"  Key: {song_plan.key}")
        print(f"  BPM: {song_plan.bpm}")
        print(f"  Structure sections: {len(song_plan.structure) if song_plan.structure else 0}")

        if song_plan.structure:
            print("  Sections:")
            for i, section in enumerate(song_plan.structure):
                print(f"    {i + 1}. {section.section_name}")

        print("\n✅ Basic system test completed!")

    except Exception as e:
        print(f"❌ Error in basic test: {e}")
        import traceback
        traceback.print_exc()


def try_with_chord_generation():
    """Test with chord generation if available"""
    print("\n🎼 Testing Chord Generation")
    print("=" * 40)

    try:
        # This would require implementing the ChordGenerator class
        # For now, let's show a simulated version

        print("Simulating chord generation...")

        # Basic chord progression simulation
        chord_progressions = [
            {
                "section_name": "Verse",
                "chords": ["Am", "F", "C", "G"],
                "key": "C major"
            },
            {
                "section_name": "Chorus",
                "chords": ["F", "G", "Am", "F"],
                "key": "C major"
            }
        ]

        print("Generated chord progressions:")
        for prog in chord_progressions:
            print(f"  {prog['section_name']}: {' - '.join(prog['chords'])}")

        print("✅ Chord generation simulation completed!")

    except Exception as e:
        print(f"❌ Error in chord test: {e}")


def simulate_iterative_workflow():
    """Simulate the iterative workflow"""
    print("\n🔄 Simulating Iterative Workflow")
    print("=" * 40)

    stages = [
        "Planning",
        "Chord Generation",
        "Lyrics Generation",
        "Melody Generation",
        "Arrangement"
    ]

    for i, stage in enumerate(stages):
        print(f"\nStage {i + 1}: {stage}")

        # Simulate iterations
        for iteration in range(1, 3):  # Max 2 iterations per stage
            print(f"  Iteration {iteration}...")

            # Simulate some processing time
            import time
            time.sleep(0.5)

            # Simulate evaluation
            quality_score = 0.6 + (iteration * 0.2)  # Improve with iterations

            if quality_score >= 0.8:
                print(f"  ✅ {stage} approved (quality: {quality_score:.2f})")
                break
            else:
                print(f"  🔄 Refining {stage} (quality: {quality_score:.2f})")
        else:
            print(f"  ⚠️ Using best result after max iterations")

    overall_quality = 0.75
    print(f"\n🎯 Overall Composition Quality: {overall_quality:.2f}")
    print("✅ Iterative workflow simulation completed!")


def show_next_steps():
    """Show what to implement next"""
    print("\n🚀 NEXT STEPS FOR IMPLEMENTATION")
    print("=" * 50)

    steps = [
        "1. Save the ChordGenerator class to app/agents/ChordGenerator.py",
        "2. Save the IterativeCompositionWorkflow to app/workflows/iterative_composition.py",
        "3. Enhance Dorthy agent to use vector store for lyrics generation",
        "4. Enhance Nancarrow agent to generate actual MIDI sequences",
        "5. Enhance Martin agent to create audio arrangements",
        "6. Add more sophisticated evaluation functions",
        "7. Create a web interface or CLI for interactive composition"
    ]

    for step in steps:
        print(f"  {step}")

    print(f"\n📁 Key files to create/modify:")
    files = [
        "app/objects/chord_progression.py",
        "app/agents/ChordGenerator.py",
        "app/workflows/iterative_composition.py",
        "app/agents/Andy.py (enhance with chord generation)",
        "app/agents/Dorthy.py (enhance lyrics processing)",
        "app/agents/Nancarrow.py (enhance MIDI generation)",
        "app/agents/Martin.py (enhance audio processing)"
    ]

    for file in files:
        print(f"  📄 {file}")


def main():
    """Main test runner"""
    print("🎵 RAINBOW AGENT ITERATIVE SYSTEM TEST")
    print("=" * 60)

    # Test current system
    try_basic_system()

    # Test chord generation simulation
    try_with_chord_generation()

    # Simulate iterative workflow
    simulate_iterative_workflow()

    # Show next steps
    show_next_steps()

    print(f"\n🎉 All tests completed!")
    print("Ready to implement the iterative system!")


if __name__ == "__main__":
    main()
