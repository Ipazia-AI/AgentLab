from agentlab.agents.htn_agent.agent import HTNController, build_default_planner
from agentlab.agents.htn_agent.belief import BeliefState, Element
from agentlab.agents.htn_agent.predicates import pred
from agentlab.agents.htn_agent.htn import Task


def test_belief_extracts_actions_and_fields():
    obs = {
        "axtree_txt": "[a1] button New Incident\n[a2] textbox Short description",
        "pruned_html": "<label>Priority</label><label>Assignee</label>",
        "last_action_error": "",
    }
    belief = BeliefState()
    belief.update(obs)

    assert belief.has(pred("ActionAvailable", "button", "New Incident", "a1"))
    assert belief.has(pred("FieldLabel", "Priority"))
    assert belief.has(pred("FieldLabel", "Assignee"))


def test_htn_create_flow_selects_new_button():
    belief = BeliefState()
    belief.ensure_goal("Create a new incident with Priority: High.")
    belief.elements = [Element(bid="a1", role="button", name="New")]

    planner = build_default_planner()
    controller = HTNController(planner)
    controller.seed(Task("SolveTask"))

    result = controller.step(belief, obs={})
    assert result.action == 'click("a1")'
