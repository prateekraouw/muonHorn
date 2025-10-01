#include "ActionInitialization.hh"
#include "PrimaryGeneratorAction.hh"
#include "RunAction.hh"
#include "EventAction.hh"
#include "SteppingAction.hh"

ActionInitialization::ActionInitialization()
 : G4VUserActionInitialization()
{}

ActionInitialization::~ActionInitialization()
{}

void ActionInitialization::BuildForMaster() const
{
  SetUserAction(new RunAction());
}

void ActionInitialization::Build() const
{
  // Create and set RunAction for each worker thread (must be first)
  SetUserAction(new RunAction());
  
  // Set primary generator
  SetUserAction(new PrimaryGeneratorAction());
  
  // Create and set EventAction
  EventAction* eventAction = new EventAction();
  SetUserAction(eventAction);
  
  // Create and set SteppingAction
  SteppingAction* steppingAction = new SteppingAction(eventAction);
  SetUserAction(steppingAction);
}