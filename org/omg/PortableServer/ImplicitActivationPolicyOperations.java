package org.omg.PortableServer;


/**
* org/omg/PortableServer/ImplicitActivationPolicyOperations.java .
* Generated by the IDL-to-Java compiler (portable), version "3.2"
* from /Users/tester/jenkins/workspace/zulu8-build-macos-aarch64/zulu-src/corba/src/share/classes/org/omg/PortableServer/poa.idl
* Thursday, July 15, 2021 8:28:49 AM PDT
*/


/**
	 * This policy specifies whether implicit activation 
	 * of servants is supported in the created POA.
	 */
public interface ImplicitActivationPolicyOperations  extends org.omg.CORBA.PolicyOperations
{

  /**
  	 * specifies the policy value
  	 */
  org.omg.PortableServer.ImplicitActivationPolicyValue value ();
} // interface ImplicitActivationPolicyOperations
